# SPDX-License-Identifier: Apache-2.0
"""Tests for the chunk/position split behind the ``mp.tokens`` event.

The positions are the store path's only unreconstructible output — chunk
hashes are prefix-chained, so a consumer cannot derive where a chunk sat
in the sequence. They must match what the local blend matcher records
for the same chunk (``position_offset + i * chunk_size``).
"""

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    split_token_bindings,
)

CHUNK = 4


def _tokens(count: int) -> list[int]:
    return list(range(100, 100 + count))


def test_offsets_are_absolute_positions_of_each_chunk():
    """A store of the whole sequence reports every complete chunk at its
    own absolute position."""
    chunks, offsets = split_token_bindings(_tokens(12), 0, 12, CHUNK)

    assert offsets == [0, 4, 8]
    assert chunks == [
        [100, 101, 102, 103],
        [104, 105, 106, 107],
        [108, 109, 110, 111],
    ]


def test_offsets_start_at_the_stored_range_not_at_zero():
    """A mid-sequence store (a blend doc range) reports positions in the
    sequence, so re-RoPE can shift from the right source position."""
    chunks, offsets = split_token_bindings(_tokens(12), 4, 12, CHUNK)

    assert offsets == [4, 8]
    assert chunks == [
        [104, 105, 106, 107],
        [108, 109, 110, 111],
    ]


def test_chunks_and_offsets_stay_parallel():
    """The two lists are consumed by a strict zip alongside the chunk
    hashes, so they must always be equal length."""
    chunks, offsets = split_token_bindings(_tokens(10), 0, 10, CHUNK)

    assert len(chunks) == len(offsets) == 2


def test_trailing_partial_chunk_is_not_reported():
    """Only complete chunks have stored KV to bind to."""
    chunks, offsets = split_token_bindings(_tokens(7), 0, 7, CHUNK)

    assert offsets == [0]
    assert chunks == [[100, 101, 102, 103]]


def test_no_complete_chunk_yields_nothing():
    assert split_token_bindings(_tokens(3), 0, 3, CHUNK) == ([], [])


def test_end_bounds_the_range_below_the_token_count():
    """``end`` short of the sequence trims the range, so a store of part
    of a request does not claim chunks it did not write."""
    chunks, offsets = split_token_bindings(_tokens(12), 0, 8, CHUNK)

    assert offsets == [0, 4]
    assert chunks[-1] == [104, 105, 106, 107]


@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_offsets_step_by_the_configured_chunk_size(chunk_size: int):
    chunks, offsets = split_token_bindings(_tokens(8), 0, 8, chunk_size)
    num_chunks = 8 // chunk_size

    assert offsets == [i * chunk_size for i in range(num_chunks)]
    assert all(len(c) == chunk_size for c in chunks)
