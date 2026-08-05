# SPDX-License-Identifier: Apache-2.0
"""Content fingerprint index over the key directory's token bindings.

Answers the blend-style **fragment** lookup: given a request's tokens,
which cached chunks does it contain, and where? Discovery is a strided
rolling-hash probe (the algorithm the local ``BlendTokenRangeMatcherV3``
uses); every candidate is then **verified token-exact** against the
binding's content, so a hash collision costs a skipped candidate rather
than a wasted prefetch.

The probe is a two-stage lookup: one vectorized gather through an
**occupancy filter** (a byte per slot, set where any fingerprint lands)
rejects the overwhelming majority of query positions, then the few
survivors resolve **exactly** through a fingerprint → content dict.
Because the filter stores no identity, two fingerprints sharing a slot
both reach the dict, which resolves each correctly — so unlike a
direct-address table that maps a slot to one entry, no indexed content
can be hidden by another and **recall is complete**. A shared slot costs
one dict miss.

The index is maintained by :class:`KeyDirectory` as bindings gain and
lose content, and holds a reference to each binding's ``uint32`` token
array rather than a copy. It carries **no model or salt awareness**: a
match names a ``chunk_hash``, and the querying server expands it into
object keys with *its own* model, salt, and world size — so a
cross-model or cross-tenant match simply misses at retrieve, exactly as
the local path already behaves.

Locking: the index owns its lock and never calls back into the
directory, so the only lock order is directory → index. Matches take
the index lock alone and therefore do not serialize behind event
application.

See ``docs/design/v1/mp_coordinator/content_index.md``.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
import threading

# Third Party
import numpy as np

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.token_hasher import (
    chunk_hash_windows_numba,
    rolling_hash_windows_numba,
)

logger = init_logger(__name__)

# Fleet-constant polynomial base. Both sides of the match live here, so
# this never has to agree with anything a server computes; it is fixed so
# stored and probed fingerprints align within a coordinator's lifetime.
POLY_BASE = np.uint64(0x9E3779B97F4A7C15)

# Filter size = smallest power of two >= _TABLE_GROWTH * live contents.
# The load factor is the filter's false-positive rate (a wasted dict
# lookup on a query position), so it is kept low — at one byte per slot
# this costs ~16 bytes per indexed chunk.
_TABLE_GROWTH = 16
_MIN_TABLE_SIZE = 1 << 10


@dataclass(frozen=True)
class ContentMatch:
    """One cached chunk found inside a query sequence.

    Attributes:
        chunk_hash: The matched chunk's ``ObjectKey.chunk_hash``. The
            caller expands it into object keys with its own model, salt,
            and world size.
        old_st: The chunk's token position in the sequence it was stored
            under (the re-RoPE source position).
        cur_st: The token position in the query where the content was
            found (the re-RoPE target position).
    """

    chunk_hash: bytes
    old_st: int
    cur_st: int


@dataclass(frozen=True)
class ContentIndexStats:
    """A point-in-time summary of index contents.

    Attributes:
        num_contents: Distinct chunk contents indexed.
        num_chunks: Chunks indexed across those contents (chunks sharing
            identical content share one entry).
        table_size: Slots in the direct-address table.
    """

    num_contents: int
    num_chunks: int
    table_size: int


@dataclass
class _Entry:
    """One distinct content: the tokens to verify against, plus every
    chunk holding that content (usually exactly one).

    ``token_ids`` is the *first* indexed chunk's content. A later chunk
    colliding on the 64-bit fingerprint with different content is not
    added, so it stays undiscoverable — never wrongly matched.
    """

    token_ids: np.ndarray
    # (chunk_hash, token_offset); the offset is per chunk, since identical
    # content can be stored at different positions.
    occupants: list[tuple[bytes, int]] = field(default_factory=list)


class ContentIndex:
    """Thread-safe content fingerprint index for fragment lookups.

    Mutations arrive through :meth:`add` and :meth:`remove` (driven by
    binding lifecycle); reads through :meth:`match` and :meth:`stats`.
    """

    def __init__(self, chunk_size: int = 256, probe_stride: int = 1) -> None:
        """Initialize an empty index.

        Args:
            chunk_size: Tokens per indexed chunk, and the match window.
                Must equal the fleet's chunk size; content of any other
                length is not indexable.
            probe_stride: Query positions between probes. ``1`` probes
                every offset for full recall; raise only to trade recall
                for CPU.

        Raises:
            ValueError: If ``chunk_size`` or ``probe_stride`` is < 1.
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1 (got {chunk_size})")
        if probe_stride < 1:
            raise ValueError(f"probe_stride must be >= 1 (got {probe_stride})")
        self._chunk_size = chunk_size
        self._probe_stride = probe_stride
        self._lock = threading.Lock()
        # Exact resolution: fingerprint -> content. Authoritative.
        self._contents: dict[int, _Entry] = {}
        # Occupancy filter: 1 where some fingerprint lands. Deliberately
        # carries no identity, so a bucket shared by two fingerprints
        # simply admits both to the dict lookup instead of hiding one.
        self._slots = np.zeros(_MIN_TABLE_SIZE, dtype=np.uint8)
        self._mask = np.uint64(_MIN_TABLE_SIZE - 1)
        # Bits currently set; a removal leaves its bit behind (a stale bit
        # only costs a dict miss), so rebuild once they outgrow the entries.
        self._bits_set = 0
        # Logged once: a fleet chunk-size disagreement would otherwise warn
        # on every store.
        self._warned_bad_length = False

    def add(self, token_ids: np.ndarray, chunk_hash: bytes, token_offset: int) -> None:
        """Index ``chunk_hash``'s content, or attach it to an existing entry.

        Idempotent: re-adding the same chunk under the same content is a
        no-op. Content whose length is not ``chunk_size`` is ignored (it
        can never match a ``chunk_size`` query window).

        Args:
            token_ids: The chunk's tokens. Held by reference, so it must
                not be mutated afterwards.
            chunk_hash: The chunk's ``ObjectKey.chunk_hash``.
            token_offset: The chunk's position in its stored sequence.
        """
        if token_ids.shape[0] != self._chunk_size:
            self._warn_bad_length(token_ids.shape[0])
            return
        poly = self._fingerprint(token_ids)
        with self._lock:
            entry = self._contents.get(poly)
            if entry is None:
                self._contents[poly] = _Entry(
                    token_ids=token_ids, occupants=[(chunk_hash, token_offset)]
                )
                slot = poly & int(self._mask)
                if not self._slots[slot]:
                    self._slots[slot] = 1
                    self._bits_set += 1
                if _TABLE_GROWTH * len(self._contents) > self._slots.shape[0]:
                    self._rebuild_locked()
                return
            for index, (held_hash, _) in enumerate(entry.occupants):
                if held_hash == chunk_hash:
                    entry.occupants[index] = (chunk_hash, token_offset)
                    return
            entry.occupants.append((chunk_hash, token_offset))

    def remove(self, token_ids: np.ndarray, chunk_hash: bytes) -> None:
        """Drop ``chunk_hash`` from the entry for ``token_ids``.

        The content itself is dropped once its last chunk leaves.
        Removing an unknown chunk or content is a no-op.

        Args:
            token_ids: The content the chunk was indexed under.
            chunk_hash: The chunk to drop.
        """
        if token_ids.shape[0] != self._chunk_size:
            return
        poly = self._fingerprint(token_ids)
        with self._lock:
            entry = self._contents.get(poly)
            if entry is None:
                return
            entry.occupants = [
                occupant for occupant in entry.occupants if occupant[0] != chunk_hash
            ]
            if entry.occupants:
                return
            del self._contents[poly]
            # The bit stays until a rebuild: clearing it here could hide a
            # different fingerprint sharing the bucket.
            if self._bits_set > 2 * len(self._contents):
                self._rebuild_locked()

    def match(self, tokens: np.ndarray) -> list[ContentMatch]:
        """Find indexed chunks contained in ``tokens``.

        Rolls a ``chunk_size`` window hash over the query, filters every
        ``probe_stride``-th position in one gather, resolves the
        survivors exactly, then **verifies each candidate token-exact**
        before accepting it.

        Args:
            tokens: The query token ids (any dtype castable to
                ``uint64``).

        Returns:
            Matches in ascending ``cur_st`` order, at most one per
            chunk; empty when nothing matched.
        """
        query = np.asarray(tokens, dtype=np.uint64)
        if query.shape[0] < self._chunk_size:
            return []
        rolling = rolling_hash_windows_numba(query, self._chunk_size, POLY_BASE)
        probe = rolling[:: self._probe_stride]
        window = self._chunk_size
        matches: list[ContentMatch] = []
        seen: set[bytes] = set()
        with self._lock:
            # One gather through the occupancy filter, then exact dict
            # resolution on the few surviving positions. The filter never
            # discriminates between fingerprints sharing a bucket, so no
            # indexed content can be hidden by another — recall is complete.
            occupied = self._slots[probe & self._mask]
            for position in np.nonzero(occupied)[0].tolist():
                entry = self._contents.get(int(probe[position]))
                if entry is None:
                    continue  # bucket shared with another fingerprint
                cur_st = position * self._probe_stride
                if not np.array_equal(query[cur_st : cur_st + window], entry.token_ids):
                    continue  # fingerprint collision: content differs
                for chunk_hash, token_offset in entry.occupants:
                    if chunk_hash in seen:
                        continue
                    seen.add(chunk_hash)
                    matches.append(
                        ContentMatch(
                            chunk_hash=chunk_hash,
                            old_st=token_offset,
                            cur_st=cur_st,
                        )
                    )
                    break  # occupants are content-identical; one suffices
        return matches

    def stats(self) -> ContentIndexStats:
        """Return a point-in-time summary of index contents.

        Returns:
            Distinct contents, total chunks, and the table size.
        """
        with self._lock:
            return ContentIndexStats(
                num_contents=len(self._contents),
                num_chunks=sum(
                    len(entry.occupants) for entry in self._contents.values()
                ),
                table_size=int(self._slots.shape[0]),
            )

    # -- Internals -------------------------------------------------------------

    def _fingerprint(self, token_ids: np.ndarray) -> int:
        """Return the 64-bit polynomial fingerprint of one chunk's content."""
        window = np.asarray(token_ids, dtype=np.uint64)
        return int(chunk_hash_windows_numba(window, self._chunk_size, POLY_BASE)[0])

    def _warn_bad_length(self, length: int) -> None:
        """Warn once that content of the wrong length cannot be indexed."""
        if self._warned_bad_length:
            return
        self._warned_bad_length = True
        logger.warning(
            "Not indexing chunk content of %d tokens: the index matches a "
            "%d-token window, so only full-chunk content is discoverable "
            "(fleet chunk-size disagreement?). Logged once.",
            length,
            self._chunk_size,
        )

    def _rebuild_locked(self) -> None:
        """Resize the occupancy filter and rebuild it from live contents,
        clearing bits left behind by removals."""
        size = _MIN_TABLE_SIZE
        while size < _TABLE_GROWTH * len(self._contents):
            size <<= 1
        self._slots = np.zeros(size, dtype=np.uint8)
        self._mask = np.uint64(size - 1)
        if self._contents:
            polys = np.fromiter(
                self._contents.keys(), dtype=np.uint64, count=len(self._contents)
            )
            self._slots[polys & self._mask] = 1
        self._bits_set = int(np.count_nonzero(self._slots))
