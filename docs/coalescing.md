<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Coalescing

hip-moi's primary contract is exact access-level instrumentation: users replace
the LDS loads and stores they actually wrote with `ctx.lds_load` and
`ctx.lds_store`, and diagnostics are based on those exact access records.

Coalescing is an optimization direction layered on top of that exact model. A
coalesced access is a compact summary of multiple exact access records that the
detector has proven to follow a regular pattern. The current implementation
records coalescing summaries as metadata only; exact access records still drive
diagnostics.

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

## Current Overhead

### No Opt-In

When no access passes a nonzero site id, hip-moi still logs exact access records
as before. Each record carries a `site_id` field containing zero.

With the default host/static storage paths, thread-level `ctx.syncthreads()`
also runs a small summary pass over the just-ended epoch. In the no-opt-in case,
that pass observes that every site id is zero and emits no summaries. The
per-access behavior remains exact logging; the extra work is an epoch-boundary
scan, performed by one thread, over the exact records in that epoch.

Users who provide their own `storage_ref` may leave the coalesced summary
buffer pointers null. In that case the summary pass returns immediately, and no
coalescing metadata is produced.

### Opted-In Sites

For a nonzero site id, hip-moi still records exact accesses first. At
`ctx.syncthreads()`, thread-level mode scans the just-ended epoch and tries to
prove that records from the same site form a regular pattern.

Today, opting in therefore costs:

* the same exact per-access logging as before,
* a nonzero `site_id` field in those exact records,
* an epoch-boundary proof pass over exact records,
* one coalesced summary record for each proven regular site, if summary storage
  has capacity.

The current implementation does not yet reduce the hot-path access-log traffic.
It is intentionally a summary/opportunity detector first. Future work may use
these proofs to compress logging, but that is not implemented yet.

## Representation

Thread-level summaries use `thread_level_context::coalesced_access_record`.
Conceptually, one summary says:

> In this epoch, a contiguous range of threads in one subgroup all performed the
> same kind of access, from the same static site, with the same byte size, and
> their addresses followed a fixed stride.

The fields are:

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

Subgroup-level context already has a separate `coalesced_access_record` type in
its public storage shape, but subgroup-level mode does not emit summaries yet.
Because subgroup-level hot access records intentionally omit thread ids,
subgroup-level coalescing needs separate proof metadata before it can be made
meaningful.

## Patterns Detected Today

The current automatic coalescing code is deliberately narrow.

It only runs in `thread_level_context`, and only at `ctx.syncthreads()`. It can
emit a summary when all of the following are true for a group of exact records:

* the records have a nonzero `site_id`,
* they are in the same epoch,
* they are in the same subgroup,
* they have the same access kind,
* they have the same `byte_count`,
* their thread ids form one contiguous range,
* each participating thread has exactly one record for that site in that epoch,
* their addresses follow one fixed signed byte stride,
* the stride magnitude is at least `byte_count`, so the represented accesses do
  not overlap,
* the represented byte span fits in the current summary field.

The current code does not summarize:

* default `site_id == 0` accesses,
* repeated dynamic instances of the same site by the same thread in one epoch,
* irregular address expressions,
* tile shapes that need more than one affine stride,
* patterns crossing subgroup boundaries,
* subgroup-level access records,
* accesses that are not visible to hip-moi because they were raw LDS loads or
  stores instead of `ctx.lds_load` / `ctx.lds_store`.

This means coalescing is currently most useful as a way to observe simple,
regular all-thread LDS access sites. It is not yet a performance feature users
should rely on to reduce instrumentation overhead.
