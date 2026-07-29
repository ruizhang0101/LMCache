# Cache-event emission (MP server → key directory)

Module: `lmcache/v1/mp_coordinator/cache_events.py`
Contract vocabulary: `lmcache/v1/mp_coordinator/api.py`
Consumer: `lmcache/v1/mp_coordinator/key_directory.py` (see
[key_directory.md](key_directory.md))

This is the emission half of the key directory (M1 of the control-plane
RFC, [issue #4226](https://github.com/LMCache/LMCache/issues/4226)): MP
servers turn storage-listener callbacks into `CacheEventBatch` streams
and deliver them to the coordinator's directory.

## The transport seam

Production deployments may replace direct HTTP push with Kafka or
another message queue. The design isolates that choice behind one
interface so nothing else changes:

```
storage listeners ──► CacheEventEmitter ──► CacheEventSink ──► directory
   (what happened)     (seq, batching)       (transport)
```

- **`CacheEventSink`** — `publish(batches)` with **at-least-once**
  delivery, preserving list order within and across calls. That is the
  entire transport contract, and it is deliberately weak: the directory
  already absorbs everything a real transport does wrong. Redelivery is
  deduplicated by the per-instance `seq` cursor, loss surfaces as a
  `seq` gap that flags the instance for resync, and restarts are fenced
  by `incarnation`. A sink never needs exactly-once or global ordering.
- **`HttpCacheEventSink`** — the first sink: one
  `POST /directory/events` per flush, batches in list order. Failures
  raise `CacheEventPublishError`; the caller decides retry vs drop
  (both are safe, see above).
- A future **Kafka sink** produces to a topic with the message key set
  to `instance_id`, so one partition carries one instance's stream —
  partition FIFO is exactly the per-instance FIFO the directory needs.
  The coordinator side gains a consumer that feeds
  `KeyDirectory.apply_batch`; the emitter and listeners are untouched.

## CacheEventEmitter

One emitter per MP-server process. Producers call
`record(event_type, tier, backend, entries)` from arbitrary threads; a
single asyncio task flushes on a timer (`run()`), so flushes are
naturally serialized.

- **Order-preserving batching.** The buffer is a list of *runs*:
  consecutive records with the same `(event_type, tier, backend)`
  identity coalesce into one pending batch; an identity change starts a
  new run. Flushing emits one `CacheEventBatch` per run, so the batch
  sequence preserves the total order of recorded events — a store
  followed by a delete of the same key can never be reordered into
  "delete, then store". (Grouping by type across the whole window would
  break exactly that case.)
- **`seq` is consumed even when publish fails.** A failed flush drops
  the drained list (bounding memory while the coordinator is down) but
  keeps the `seq` numbers it assigned. The directory sees a gap and
  sets `gap_detected` for the instance — the honest signal that events
  were lost and the resync backstop (future work) should reconcile.
  Reusing the seqs instead would hide partial-delivery ambiguity (an
  HTTP timeout after the coordinator applied the batch).
- **`incarnation` = server start time** (`int(time.time())` at
  lifespan startup). A restarted server's first batch fences out every
  placement its previous incarnation reported, matching the fact that
  its pools restarted empty.

## Listeners

`L2DirectoryListener` implements `L2AdapterListener` and forwards
stored/accessed/deleted callbacks as `STORE`/`ACCESS`/`DELETE` events at
`tier=l2`. The listener callbacks do not identify the emitting adapter,
so the HTTP-server lifespan registers **one listener per adapter**,
each bound to that adapter's backend name (`AdapterDescriptor.type_name`
via `StorageManager.l2_adapters()`).

`L1DirectoryListener` implements `L1ManagerListener` (registered via
`StorageManager.register_l1_listener`) and maps at `tier=l1`:

| callback | event |
| --- | --- |
| `on_l1_keys_write_finished`, `on_l1_keys_finish_write_and_reserve_read` (prefetch) | `STORE` |
| `on_l1_keys_deleted_by_manager` (evictions included) | `DELETE` |
| `on_l1_keys_read_finished`, `on_l1_keys_accessed` | `ACCESS` |
| `on_l1_keys_reserved_read` / `on_l1_keys_reserved_write` | ignored (reservations are not state changes) |

The two store-side L1 callbacks carry `sizes` (added alongside this
module, mirroring `on_l2_keys_stored`), reported from
`MemoryObj.get_size()` at write-finish. The L1 backend name is derived
from the configured medium by `l1_backend_name`: `"gds"`, `"devdax"`,
or `"dram"`.

**Multi-medium L1** (hybrid DRAM + Device-DAX overflow, where DRAM
fills first and allocations spill into DAX) is a single label today —
the L1 listener callbacks don't say which medium an object landed in.
When per-medium placements matter, the L1 manager must report the
medium per key (alongside `sizes` at write-finish); the listener then
splits entries into per-backend `record()` calls. The emitter and
directory need no changes — `backend` is already per-record, and the
directory keys placements by `(instance, tier, backend)`.

## Wiring and configuration

Enabled in the MP HTTP server lifespan when a coordinator URL is set
and `--coordinator-event-reporting` (or
`LMCACHE_COORDINATOR_EVENT_REPORTING`) is on;
`--coordinator-event-flush-interval` tunes the flush timer (default
1s).

This stream is the MP server's **only** event feed to the coordinator:
on the coordinator, `/directory/events` applies each batch to the key
directory, and batches the directory *applies* (past seq dedup and
incarnation fencing) fan out through `CacheEventRouter`
(`cache_control/event_router.py`) to the L2 usage ledger and eviction
LRU. The legacy `/quota/events` endpoint, its `UsageEvent` schema, and
the MP-server `L2EventListener` are gone. Two deliberate semantics in
the router:

- **Replay-safe quota**: consumers only see directory-applied batches,
  so redelivered batches can't double-count usage (the legacy stream
  had no dedup).
- **No incarnation fencing for quota**: fencing drops *placements*
  (an MP restart empties its pools), but L2 bytes on shared/durable
  backends survive restarts, so the usage ledger is keyed by
  `ObjectKey` and only shrinks on `DELETE` events.

## Known limitations (follow-ups)

- **Runtime-added L2 adapters** (`add_l2_adapter`) do not get a
  directory listener; their placements are invisible until a
  listener-factory registration or the resync backstop exists.
- **No final flush on shutdown**: buffered events at shutdown are
  lost; deregistration's `drop_instance` and incarnation fencing on
  restart make this benign.
- **Multi-medium L1 attribution** (see the listeners section): per-key
  medium reporting from the L1 manager, needed for hybrid DRAM+DAX.
