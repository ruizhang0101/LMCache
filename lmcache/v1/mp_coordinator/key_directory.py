# SPDX-License-Identifier: Apache-2.0
"""Fleet-wide key directory for the MP coordinator.

Maps each :class:`ObjectKey` to its placements (instance, tier, backend,
size) across the fleet, using :class:`CacheEventBatch` streams from MP
servers. The directory is eventually consistent: lookups are hints to be
validated at the owner.

Events are processed in order per instance and only the latest L1
placements for each incarnation are kept. L2 placements persist across
restarts.

Views like per-``cache_salt`` L2 usage are maintained separately from
the same event stream.

See ``docs/design/v1/mp_coordinator/key_directory.md``.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from enum import Enum
import threading

# Third Party
import numpy as np

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    UNKNOWN_TOKEN_OFFSET,
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.content_index import (
    ContentIndex,
    ContentIndexStats,
    ContentMatch,
)

logger = init_logger(__name__)

# Token ids are held as ``uint32``: a few hundred bytes per chunk instead
# of the ~10 KB a ``tuple[int, ...]`` of boxed ints costs, and content
# comparison against a query window stays vectorized.
_TOKEN_DTYPE = np.uint32

# Shared read-only empty array for chunks whose content is unknown.
_NO_TOKENS = np.empty(0, dtype=_TOKEN_DTYPE)
_NO_TOKENS.flags.writeable = False


@dataclass(frozen=True)
class Placement:
    """One live placement of a key, as returned by directory lookups.

    Attributes:
        instance_id: The emitter that most recently reported the placement.
        incarnation: The reporting instance's incarnation at report time.
        tier: Tier the bytes live on (``l1`` or ``l2``).
        backend: Backend within the tier.
        size_bytes: Size the owner reported at store time.
        shared: ``True`` when the backend is a fleet-shared pool (see
            :class:`CacheEventBatch`).
    """

    instance_id: str
    incarnation: int
    tier: Tier
    backend: str
    size_bytes: int
    shared: bool = False


class ApplyResult(str, Enum):
    """Result of applying one :class:`CacheEventBatch` to the directory.

    ``APPLIED`` — the batch was applied.
    ``DUPLICATE`` — the batch's ``seq`` was already applied for the
    instance's current incarnation; the batch was dropped.
    ``STALE_INCARNATION`` — the batch carries an incarnation older than the
    instance's current one; the batch was dropped.
    """

    APPLIED = "applied"
    DUPLICATE = "duplicate"
    STALE_INCARNATION = "stale_incarnation"


@dataclass(frozen=True)
class InstanceDirectoryStats:
    """Directory-side bookkeeping for one reporting instance.

    Attributes:
        incarnation: The instance's current incarnation.
        last_seq: Highest batch ``seq`` applied for that incarnation.
        gap_detected: ``True`` if a ``seq`` gap was observed for the
            instance's stream.
        num_l1_keys: Number of keys the stream has reported L1
            placements for (eventually consistent).
    """

    incarnation: int
    last_seq: int
    gap_detected: bool
    num_l1_keys: int


@dataclass(frozen=True)
class DirectoryStats:
    """A point-in-time summary of directory contents.

    Attributes:
        num_keys: Keys with at least one placement.
        num_placements: Total placements across all keys.
        instances: Per-instance bookkeeping, keyed by ``instance_id``.
        content: Content-index counts — how much of the directory is
            fragment-matchable (see :class:`ContentIndexStats`).
    """

    num_keys: int
    num_placements: int
    instances: dict[str, InstanceDirectoryStats]
    content: ContentIndexStats


@dataclass
class _KeyRecord:
    """Directory value for one key: its placements plus recency."""

    placements: list[Placement] = field(default_factory=list)
    last_access: float = 0.0


@dataclass(frozen=True)
class TokenBinding:
    """One chunk's known token content, as returned by the directory.

    Attributes:
        token_ids: The chunk's token ids as a read-only ``uint32`` array;
            empty when the directory does not know the chunk's content.
        token_offset: Token position of the chunk's first token in the
            sequence it was stored under, or
            :data:`~lmcache.v1.mp_coordinator.api.UNKNOWN_TOKEN_OFFSET`
            when no emitter has reported one (including for chunks whose
            content is unknown).
    """

    token_ids: np.ndarray
    token_offset: int


@dataclass
class _TokenBinding:
    """Token content for one chunk hash plus the keys sharing it (dropped
    when the last key goes). ``token_ids`` is empty until a
    token-bearing ``STORE`` entry arrives."""

    token_ids: np.ndarray
    token_offset: int
    keys: set[ObjectKey]


@dataclass
class _InstanceState:
    """Per-instance event-stream cursor and reverse index."""

    incarnation: int
    last_seq: int = 0
    gap_detected: bool = False
    keys: set[ObjectKey] = field(default_factory=set)


class KeyDirectory:
    """Thread-safe in-memory key directory built from cache events.

    Mutations arrive through :meth:`apply_batch` and :meth:`drop_instance`;
    reads through :meth:`lookup` and :meth:`stats`. Nothing is persisted.
    """

    def __init__(self, chunk_size: int = 256, probe_stride: int = 1) -> None:
        """Initialize an empty directory and its content index.

        Args:
            chunk_size: Fleet chunk size; the content index's match
                window.
            probe_stride: Query positions between content-index probes.

        Raises:
            ValueError: If ``chunk_size`` or ``probe_stride`` is < 1.
        """
        self._lock = threading.Lock()
        self._records: dict[ObjectKey, _KeyRecord] = {}
        self._instances: dict[str, _InstanceState] = {}
        # chunk hash → tokens + keys, for chunk hashes of >= 1 record.
        self._token_bindings: dict[bytes, _TokenBinding] = {}
        # Derived from the bindings; owns its own lock (order: self → index).
        self._content_index = ContentIndex(
            chunk_size=chunk_size, probe_stride=probe_stride
        )

    def apply_batch(self, batch: CacheEventBatch) -> ApplyResult:
        """Apply one event batch to the directory.

        Applies incarnation fencing, seq dedup, and gap detection, then
        the entries. Entry application is idempotent: re-storing upserts
        the placement (and its token binding), deleting an absent
        placement is a no-op.

        Args:
            batch: The event batch to apply.

        Returns:
            Whether the batch was applied, or why it was dropped.
        """
        with self._lock:
            state = self._instances.get(batch.instance_id)
            if state is None:
                state = _InstanceState(incarnation=batch.incarnation)
                self._instances[batch.instance_id] = state
            elif batch.incarnation < state.incarnation:
                return ApplyResult.STALE_INCARNATION
            elif batch.incarnation > state.incarnation:
                # Restart: fence out the previous incarnation's placements.
                self._drop_instance_locked(batch.instance_id)
                state = _InstanceState(incarnation=batch.incarnation)
                self._instances[batch.instance_id] = state
            elif batch.seq <= state.last_seq:
                return ApplyResult.DUPLICATE

            if batch.seq > state.last_seq + 1 and not state.gap_detected:
                state.gap_detected = True
                logger.warning(
                    "Event gap for instance %s (incarnation %d): "
                    "seq jumped %d -> %d; slice needs replay",
                    batch.instance_id,
                    batch.incarnation,
                    state.last_seq,
                    batch.seq,
                )
            state.last_seq = batch.seq

            for entry in batch.entries:
                self._apply_entry_locked(state, batch, entry)
            return ApplyResult.APPLIED

    def lookup(self, keys: list[ObjectKey]) -> list[list[Placement]]:
        """Return the known placements for each requested key.

        Args:
            keys: The keys to look up.

        Returns:
            One placement list per requested key, in request order —
            empty for unknown keys. Each list is sorted by
            ``(instance_id, tier, backend)``.
        """
        with self._lock:
            results: list[list[Placement]] = []
            for key in keys:
                record = self._records.get(key)
                if record is None:
                    results.append([])
                    continue
                results.append(
                    sorted(
                        record.placements,
                        key=lambda p: (p.instance_id, p.tier.value, p.backend),
                    )
                )
            return results

    def get_token_bindings(self, chunk_hashes: list[bytes]) -> list[TokenBinding]:
        """Return the known token content for each requested chunk hash.

        Args:
            chunk_hashes: ``ObjectKey.chunk_hash`` values to look up.

        Returns:
            One binding per hash, in request order. Unknown chunks — and
            chunks no token-bearing entry has arrived for — yield a
            binding with empty ``token_ids``.
        """
        with self._lock:
            results: list[TokenBinding] = []
            for chunk_hash in chunk_hashes:
                binding = self._token_bindings.get(chunk_hash)
                if binding is None:
                    results.append(
                        TokenBinding(
                            token_ids=_NO_TOKENS,
                            token_offset=UNKNOWN_TOKEN_OFFSET,
                        )
                    )
                else:
                    results.append(
                        TokenBinding(
                            token_ids=binding.token_ids,
                            token_offset=binding.token_offset,
                        )
                    )
            return results

    def match_content(self, tokens: np.ndarray) -> list[ContentMatch]:
        """Find cached chunks contained anywhere in ``tokens``.

        The fragment lookup behind fleet-wide CacheBlend reuse: unlike
        :meth:`lookup`, the query need not be a prefix — each match is a
        chunk of content found at some offset. Matches name a
        ``chunk_hash`` only; the caller expands it into object keys with
        its own model, salt, and world size, so a cross-model or
        cross-tenant match misses at retrieve rather than being filtered
        here.

        Runs under the content index's lock, not the directory's, so it
        does not serialize behind event application.

        Args:
            tokens: The query token ids.

        Returns:
            Matches in ascending ``cur_st`` order, at most one per chunk.
            Matches may overlap in the query: callers that scatter them
            must resolve overlaps themselves.
        """
        return self._content_index.match(tokens)

    def content_stats(self) -> ContentIndexStats:
        """Return a point-in-time summary of the content index.

        Returns:
            Distinct contents, total chunks, and the table size.
        """
        return self._content_index.stats()

    def list_keys(
        self,
        tier: Tier = Tier.ALL,
        instance_id: str = "",
        backend: str = "",
        offset: int = 0,
        limit: int = 1000,
    ) -> tuple[int, dict[ObjectKey, list[Placement]]]:
        """List keys whose placements match the filters, one page at a time.

        A snapshot for inspection: iteration order is the directory's
        insertion order and is not stable across mutations, so pages of
        a changing directory may skip or repeat keys.

        Args:
            tier: Keep placements on this tier (``all`` keeps every tier).
            instance_id: Keep placements reported by this instance
                (empty keeps every instance).
            backend: Keep placements on this backend (empty keeps every
                backend).
            offset: Matching keys to skip.
            limit: Maximum keys to return.

        Returns:
            ``(total, page)``: the number of keys with at least one
            matching placement, and the ``[offset, offset + limit)``
            slice of them as an ordered mapping of key → its matching
            placements.

        Raises:
            ValueError: If ``offset`` or ``limit`` is negative.
        """
        if offset < 0:
            raise ValueError(f"offset must be >= 0 (got {offset})")
        if limit < 0:
            raise ValueError(f"limit must be >= 0 (got {limit})")
        with self._lock:
            total = 0
            page: dict[ObjectKey, list[Placement]] = {}
            for key, record in self._records.items():
                placements = [
                    p
                    for p in record.placements
                    if (tier == Tier.ALL or p.tier == tier)
                    and (not instance_id or p.instance_id == instance_id)
                    and (not backend or p.backend == backend)
                ]
                if not placements:
                    continue
                if total >= offset and len(page) < limit:
                    page[key] = placements
                total += 1
            return total, page

    def drop_instance(self, instance_id: str) -> int:
        """Remove every **L1** placement reported by ``instance_id``.

        The instance's stream cursor is removed too, so a later reconnect
        starts fresh with any incarnation.

        Args:
            instance_id: The instance whose placements to drop.

        Returns:
            The number of placements removed.
        """
        with self._lock:
            removed = self._drop_instance_locked(instance_id)
            self._instances.pop(instance_id, None)
            return removed

    def reconcile(self, batch: CacheEventBatch) -> None:
        """Apply ``batch``'s entries without stream-cursor bookkeeping.

        Args:
            batch: The synthesized batch to apply.
        """
        with self._lock:
            state = self._instances.get(batch.instance_id)
            if state is None:
                state = _InstanceState(incarnation=batch.incarnation)
                self._instances[batch.instance_id] = state
            for entry in batch.entries:
                self._apply_entry_locked(state, batch, entry)

    def stats(self) -> DirectoryStats:
        """Return a point-in-time summary of directory contents.

        Returns:
            Key/placement counts, per-instance stream state keyed by
            ``instance_id``, and the content-index counts.
        """
        content = self._content_index.stats()
        with self._lock:
            num_placements = sum(
                len(record.placements) for record in self._records.values()
            )
            instances = {
                instance_id: InstanceDirectoryStats(
                    incarnation=state.incarnation,
                    last_seq=state.last_seq,
                    gap_detected=state.gap_detected,
                    num_l1_keys=len(state.keys),
                )
                for instance_id, state in self._instances.items()
            }
            return DirectoryStats(
                num_keys=len(self._records),
                num_placements=num_placements,
                instances=instances,
                content=content,
            )

    # -- Internals (call with self._lock held) --------------------------------

    def _apply_entry_locked(
        self,
        state: _InstanceState,
        batch: CacheEventBatch,
        entry: CacheEventEntry,
    ) -> None:
        """Apply one entry of ``batch`` under the directory lock."""
        key = entry.key.to_object_key()
        if batch.event_type == CacheEventType.STORE:
            record = self._records.get(key)
            if record is None:
                record = _KeyRecord()
                self._records[key] = record
                self._link_key(key)
            placement = Placement(
                instance_id=batch.instance_id,
                incarnation=batch.incarnation,
                tier=batch.tier,
                backend=batch.backend,
                size_bytes=entry.size_bytes,
                shared=batch.shared,
            )
            index = self._find_placement(record.placements, batch)
            if index is None:
                record.placements.append(placement)
            else:
                record.placements[index] = placement
            if entry.token_ids:
                self._fill_binding_locked(key.chunk_hash, entry)
            record.last_access = max(record.last_access, batch.ts)
            if batch.tier == Tier.L1:
                state.keys.add(key)
        elif batch.event_type == CacheEventType.DELETE:
            record = self._records.get(key)
            if record is None:
                return
            index = self._find_placement(record.placements, batch)
            if index is not None:
                record.placements.pop(index)
            if not record.placements:
                del self._records[key]
                self._unlink_key(key)
            if batch.tier == Tier.L1 and not any(
                p.tier == Tier.L1 and p.instance_id == batch.instance_id
                for p in record.placements
            ):
                state.keys.discard(key)
        elif batch.event_type == CacheEventType.ACCESS:
            record = self._records.get(key)
            if record is not None:
                record.last_access = max(record.last_access, batch.ts)

    @staticmethod
    def _find_placement(
        placements: list[Placement], batch: CacheEventBatch
    ) -> int | None:
        """Return the index of the placement whose identity matches
        ``batch``, or ``None`` if absent."""
        for index, placement in enumerate(placements):
            if (
                placement.shared == batch.shared
                and (batch.shared or placement.instance_id == batch.instance_id)
                and placement.tier == batch.tier
                and placement.backend == batch.backend
            ):
                return index
        return None

    def _drop_instance_locked(self, instance_id: str) -> int:
        """Remove the **L1** placements ``instance_id`` reported; return
        the count. L2 placements survive: their bytes persist across the
        reporter's restarts and leave only via ``DELETE`` events."""
        state = self._instances.get(instance_id)
        if state is None:
            return 0
        removed = 0
        for key in state.keys:
            record = self._records.get(key)
            if record is None:
                continue
            kept = [
                p
                for p in record.placements
                if p.tier != Tier.L1 or p.instance_id != instance_id
            ]
            removed += len(record.placements) - len(kept)
            if kept:
                record.placements = kept
            else:
                del self._records[key]
                self._unlink_key(key)
        state.keys.clear()
        return removed

    def _fill_binding_locked(self, chunk_hash: bytes, entry: CacheEventEntry) -> None:
        """Record ``entry``'s token content on ``chunk_hash``'s binding.

        Token ids outside ``uint32`` leave the binding as it was — a
        lookup miss, repaired by the chunk's next well-formed entry —
        rather than failing the whole batch over one bad entry.

        An entry whose ``token_offset`` is
        :data:`~lmcache.v1.mp_coordinator.api.UNKNOWN_TOKEN_OFFSET` still
        fills the binding's content (so ``key -> tokens`` introspection
        works) but is **not** added to the content index: without the
        stored position, a fragment match could not tell the requester
        where to re-RoPE the chunk from, and a wrong position yields
        wrong KV rather than a miss.

        Args:
            chunk_hash: Chunk hash whose binding to fill.
            entry: The store entry carrying the token ids and offset.
        """
        try:
            token_ids = np.asarray(entry.token_ids, dtype=_TOKEN_DTYPE)
        except (OverflowError, TypeError, ValueError):
            logger.warning(
                "Ignoring token ids for chunk %s: values outside uint32",
                chunk_hash.hex(),
            )
            return
        token_ids.flags.writeable = False
        binding = self._token_bindings[chunk_hash]
        if binding.token_ids.size and not np.array_equal(binding.token_ids, token_ids):
            # Re-store with different content: retire the old fingerprint,
            # or the chunk stays discoverable under content it no longer has.
            self._content_index.remove(binding.token_ids, chunk_hash)
        binding.token_ids = token_ids
        binding.token_offset = entry.token_offset
        if entry.token_offset == UNKNOWN_TOKEN_OFFSET:
            return
        self._content_index.add(token_ids, chunk_hash, entry.token_offset)

    def _link_key(self, key: ObjectKey) -> None:
        """Index ``key`` under its chunk's token binding, creating an
        empty binding on first reference."""
        binding = self._token_bindings.get(key.chunk_hash)
        if binding is None:
            self._token_bindings[key.chunk_hash] = _TokenBinding(
                token_ids=_NO_TOKENS, token_offset=UNKNOWN_TOKEN_OFFSET, keys={key}
            )
        else:
            binding.keys.add(key)

    def _unlink_key(self, key: ObjectKey) -> None:
        """Remove ``key`` from its chunk's token binding, dropping the
        binding — and its content-index entry — with its last key."""
        binding = self._token_bindings.get(key.chunk_hash)
        if binding is None:
            return
        binding.keys.discard(key)
        if not binding.keys:
            del self._token_bindings[key.chunk_hash]
            if binding.token_ids.size:
                self._content_index.remove(binding.token_ids, key.chunk_hash)
