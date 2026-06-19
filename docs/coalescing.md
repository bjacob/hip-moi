<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Coalescing

hip-moi's primary contract is exact access-level instrumentation: users replace
the LDS loads and stores they actually wrote with `ctx.lds_load` and
`ctx.lds_store`. Exact records remain the correctness anchor and fallback, even
when an opted-in subgroup-level site is summarized for an epoch-close conflict
check.

Coalescing is an optimization direction layered on top of that exact model. A
coalesced access is a compact summary of multiple instrumented access records
that the detector has proven to follow a regular pattern. Thread-level
summaries are currently opportunity metadata. Subgroup-level summaries now also
participate in conflict detection at epoch boundaries when nonzero site ids and
proof-log storage are supplied.

## Opting In

By default, an access has `hip_moi::no_site_id`, whose value is zero:

```c++
ctx.lds_store(&lds[threadIdx.x], value);
int x = ctx.lds_load(&lds[threadIdx.x]);
```

That default means exact logging only. It does not opt the access site into
coalescing.

To opt in, pass a nonzero `hip_moi::site_id` as the final argument:

```c++
ctx.lds_store(&lds[threadIdx.x], value, HIP_MOI_SITE_ID());
int x = ctx.lds_load(&lds[threadIdx.x], HIP_MOI_SITE_ID());
```

`HIP_MOI_SITE_ID()` expands to a `hip_moi::site_id` built from compile-time
source-location data. It does not hide the load or store operation; the user
still calls `ctx.lds_load` or `ctx.lds_store` explicitly.

For tests or generated code, an explicit numeric id is also possible:

```c++
constexpr hip_moi::site_id tile_store_site(0x1234u);
ctx.lds_store(&lds[threadIdx.x], value, tile_store_site);
```

Prefer `HIP_MOI_SITE_ID()` in ordinary hand-written code. Use explicit numeric
ids only when the caller has a reason to manage identity itself.

For subgroup-level summaries, a nonzero site id is necessary but not sufficient:
the context also needs proof-log storage. `hip_moi::subgroup_level_host_context`
can provide that storage when requested:

```c++
hip_moi::host_context_options options;
options.access_record_capacity = 1024;
options.coalesced_access_record_capacity = 1024;
options.coalescing_proof_record_capacity = 1024;

hip_moi::subgroup_level_host_context moi(options);
```

Custom `storage_ref` users can instead populate the
`coalescing_proof_records`, `coalescing_proof_record_capacity`,
`coalescing_proof_count`, and `epoch_coalescing_proof_count` fields directly.

## Current Overhead

### No Opt-In

When no access passes a nonzero site id, hip-moi still logs exact access records
as before. Each record carries a `site_id` field containing zero.

With the default host/static storage paths, thread-level `ctx.syncthreads()`
also runs a small summary pass over the just-ended epoch. In the no-opt-in case,
that pass observes that every site id is zero and emits no summaries. The
per-access behavior remains exact logging; the extra work is an epoch-boundary
scan, performed by one thread, over the exact records in that epoch.

Subgroup-level mode does not perform pairwise conflict checks on every access.
It records exact accesses on the hot path, then checks the just-ended epoch at
`ctx.syncthreads()` or at a final uniform `ctx.finish()`. When no access opts
in, no proof records are written and no summaries are emitted, so the
epoch-close pass falls back to exact-record conflict detection for that epoch.

Users who provide their own `storage_ref` may leave the coalesced summary
buffer pointers null. In that case the summary pass returns immediately, and no
coalescing metadata is produced. For subgroup-level mode, users may also leave
the coalescing proof-log pointers null; then no subgroup summaries are produced.

### Opted-In Sites

For a nonzero site id, hip-moi still records exact accesses first. In
thread-level mode, `ctx.syncthreads()` scans the just-ended exact access records
and tries to prove that records from the same site form a regular pattern.

In subgroup-level mode, an opted-in access also writes a separate proof record
when proof storage is supplied. That proof record carries the access address and
the thread's lane within the subgroup. The subgroup-level diagnostic
`access_record` stays compact and does not grow a lane or thread-id field.

Today, opting in therefore costs:

* the same exact per-access logging as before,
* a nonzero `site_id` field in those exact records,
* in subgroup-level mode, one additional proof record per opted-in access when
  proof storage is supplied,
* an epoch-boundary proof pass over exact records or proof records,
* one coalesced summary record for each proven regular site, if summary storage
  has capacity,
* in subgroup-level mode, an epoch-boundary conflict pass over the summaries
  produced for that epoch plus exact records that were not covered by a summary.

The current implementation does not yet reduce the hot-path access-log traffic:
exact records are still recorded first. For subgroup-level mode, coalescing is
now more than metadata because proven summaries replace their covered exact
records in the epoch-close conflict pass. That is the first practical
optimization step, but it is not yet enough to make coalescing a performance
feature users should rely on for large kernels.

## Representation

Thread-level summaries use `thread_level_context::coalesced_access_record`.
Subgroup-level summaries use
`subgroup_level_context::coalesced_access_record`.

Conceptually, a thread-level summary says:

> In this epoch, a contiguous range of threads in one subgroup all performed the
> same kind of access, from the same static site, with the same byte size, and
> their addresses followed a fixed stride.

The thread-level fields are:

* `first_address`: address used by the first participating thread.
* `byte_count`: byte size of each individual access.
* `span_byte_count`: byte span covered from the lowest address to the end of
  the highest-addressed access.
* `first_thread_id`: first participating thread id.
* `subgroup_id`: subgroup containing the participating threads.
* `epoch`: shadow epoch where the accesses occurred.
* `kind`: load or store.
* `participant_count`: number of participating threads.
* `valid`: whether the summary slot is populated.
* `address_stride`: signed byte stride between consecutive participating
  threads' addresses.
* `site_id`: nonzero source-site id shared by the exact records.

Conceptually, a subgroup-level summary says:

> In this epoch, a contiguous range of lanes in one subgroup all performed the
> same kind of access, from the same static site, with the same byte size, and
> their addresses followed a fixed stride.

Its fields are the same shape, except it uses `first_lane` rather than
`first_thread_id`.

This representation can describe more than a strictly contiguous byte range.
For example, it can represent all of these shapes:

```text
thread 0: lds +  0 bytes
thread 1: lds +  4 bytes
thread 2: lds +  8 bytes
thread 3: lds + 12 bytes
```

```text
thread 0: lds +  0 bytes
thread 1: lds +  8 bytes
thread 2: lds + 16 bytes
thread 3: lds + 24 bytes
```

```text
thread 0: lds + 24 bytes
thread 1: lds + 16 bytes
thread 2: lds +  8 bytes
thread 3: lds +  0 bytes
```

The first has stride `4`, the second has stride `8`, and the third has stride
`-8`. A summary is only emitted when the stride is large enough that adjacent
participating accesses do not overlap.

## Conflict Detection

Thread-level conflict detection is still exact-record based. Thread-level
summaries are useful for observing coalescing opportunities but do not yet
change diagnostics.

Subgroup-level conflict detection has two layers:

* on every access, hip-moi appends an exact access record and, for opted-in
  sites with proof storage, an optional proof record;
* at `ctx.syncthreads()` or a final uniform `ctx.finish()`, summaries produced
  for the closing epoch are compared with each other and with exact records
  that were not themselves summarized;
* exact records that are not covered by any new summary are compared with each
  other as the fallback path.

The summary overlap check uses the represented per-lane byte ranges, not merely
the enclosing `span_byte_count`. This matters for fixed-stride patterns with
gaps: two summaries whose spans overlap do not report unless at least one
represented lane access actually overlaps.

When a subgroup-level diagnostic is emitted from a coalesced summary, the
diagnostic still uses `kind=access_conflict`. The address and size fields on a
coalesced side describe the summary's first address and enclosing span. For a
summary-vs-exact diagnostic, the exact side keeps the exact access address and
byte size.

Because subgroup-level checking is epoch-close based, kernels whose final epoch
contains accesses of interest should call `ctx.finish()` uniformly before
returning unless a trailing `ctx.syncthreads()` already closed that epoch.

## Patterns Detected Today

The current automatic coalescing code is deliberately narrow.

It only runs at `ctx.syncthreads()`. Thread-level mode proves patterns from
exact access records. Subgroup-level mode proves patterns from the separate
proof log. Both can emit a summary when all of the following are true for a
group of records:

* the records have a nonzero `site_id`,
* they are in the same epoch,
* they are in the same subgroup,
* they have the same access kind,
* they have the same `byte_count`,
* their thread ids, or subgroup lanes, form one contiguous range,
* each participating thread/lane has exactly one record for that site in that
  epoch,
* their addresses follow one fixed signed byte stride,
* the stride magnitude is at least `byte_count`, so the represented accesses do
  not overlap,
* the represented byte span fits in the current summary field.

The current code does not summarize:

* default `site_id == 0` accesses,
* repeated dynamic instances of the same site by the same thread in one epoch,
* irregular address expressions,
* tile shapes that need more than one affine stride,
* a single summary spanning multiple subgroups; subgroup-level mode emits one
  summary per subgroup,
* subgroup-level opted-in accesses when no proof storage was supplied,
* accesses that are not visible to hip-moi because they were raw LDS loads or
  stores instead of `ctx.lds_load` / `ctx.lds_store`.

This means coalescing is currently most useful as a way to observe and diagnose
simple, regular all-thread LDS access sites. It is not yet a performance
feature users should rely on to reduce instrumentation overhead.
