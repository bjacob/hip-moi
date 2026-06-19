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

Coalescing is a subgroup-level optimization layered on top of that exact model.
A coalesced access is a compact summary of multiple instrumented access records
that the detector has proven to follow a regular lane-to-address pattern.
Thread-level mode is intentionally exact-record based for now; it still records
site ids in access records and diagnostics, but it does not build coalesced
summaries.

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

For subgroup-level summaries and hot-path compression, a nonzero site id is
necessary but not sufficient: the subgroup-level context also needs coalescing
access storage. `hip_moi::subgroup_level_host_context` can provide that storage
from its byte budget:

```c++
hip_moi::host_context_options options;
options.storage_bytes = 16 * 1024 * 1024;
options.subgroup_capacity = 64;

hip_moi::subgroup_level_host_context moi(options);
```

The typed-capacity fields in `host_context_options` are advanced overrides. A
negative value asks the host context to derive the capacity from
`storage_bytes`; zero disables optional coalescing buffers. Custom `storage_ref`
users can instead populate the
`coalescing_access_records`, `coalescing_access_record_capacity`,
`coalescing_access_count`, `epoch_coalescing_access_count`, and
`coalescing_fallback_count` fields directly. They may also provide
`coalescing_group_records`,
`coalescing_group_record_capacity`, and `coalescing_group_count` scratch
storage to make subgroup-level summary construction cheaper across many
distinct sites. If group storage is absent or fills up, hip-moi falls back to
the coalescing-access scan path.

## Current Overhead

### No Opt-In

When no access passes a nonzero site id, hip-moi still logs exact access records
as before. Each record carries a `site_id` field containing zero.

Thread-level mode does not run a coalescing pass. Subgroup-level mode does not
perform pairwise conflict checks on every access. It records exact accesses on
the hot path, then checks the just-ended epoch at `ctx.syncthreads()` or at a
final uniform `ctx.finish()`. When no access opts in, no coalescing access
records are written and no summaries are emitted, so the epoch-close pass falls
back to exact-record conflict detection for that epoch.

Users who provide their own `storage_ref` may leave the coalesced summary
buffer pointers null. In that case the summary pass returns immediately, and no
coalescing metadata is produced. For subgroup-level mode, users may also leave
the coalescing access pointers null; then opted-in accesses fall back to the
ordinary exact `access_record` path and no subgroup summaries are produced.
If a `coalescing_fallback_count` counter is present, it is incremented once for
each opted-in subgroup-level access that takes this exact-record fallback path.
Default-site accesses do not increment that counter.

### Opted-In Sites

In subgroup-level mode, an opted-in access writes a
`coalescing_access_record` instead of an ordinary exact `access_record` when
coalescing access storage is supplied and has capacity. That record carries the
access address and the thread's lane within the subgroup. If coalescing access
storage is absent or full, the access falls back to the ordinary exact
`access_record` path and increments `coalescing_fallback_count` when that
counter is present. The subgroup-level diagnostic `access_record` stays compact
and does not grow a lane or thread-id field.
When group scratch storage is supplied, epoch close first builds compact
`coalescing_group_record` entries keyed by subgroup/site/kind/size in an
open-addressed scratch table. Those group records accumulate lane masks and
endpoint addresses while walking the coalescing access log once. Each group is
then validated with one coalescing-access scan for fixed-stride addresses. This
avoids both per-lane rescans and the older prior-record leader scan. When group
scratch is absent or too small, hip-moi uses the previous coalescing-access scan
path instead.

Today, subgroup-level opt-in therefore costs:

* one coalescing access record per opted-in access when coalescing access
  storage is supplied,
* fallback exact logging only when coalescing access storage is absent or full,
* optionally, one subgroup-level group scratch record per distinct
  subgroup/site/kind/size key in the closing epoch,
* an epoch-boundary summary pass over coalescing access records,
* one coalesced summary record for each proven regular site, if summary storage
  has capacity,
* an epoch-boundary conflict pass over the summaries produced for that epoch
  plus exact and coalescing access records that were not covered by a summary.

For subgroup-level opted-in accesses, coalescing now reduces ordinary exact
access-log traffic when the coalescing access log has capacity. This is still
not the final performance story: one lane-carrying coalescing access record is
written per participating lane, and epoch-close validation still has real work
to do.

The remaining known subgroup-level cost is the validation scan per candidate
group, plus the one coalescing access record written per opted-in lane. The
group scratch table is fixed-capacity and open-addressed, so unusually high
collision pressure or too little scratch capacity falls back to the older
coalescing-access scan path.

### Counters

`coalescing_access_count` counts opted-in subgroup-level accesses that entered
the coalescing access path while coalescing access storage was present. It may
exceed `coalescing_access_record_capacity`; only the first
`coalescing_access_record_capacity` records are stored.

`coalescing_fallback_count` counts opted-in subgroup-level accesses that could
not use a coalescing access record and therefore logged an ordinary exact
`access_record`. That includes the two currently observable hot-path fallback
cases: no coalescing access storage and a full coalescing access buffer.

`coalescing_group_count` counts subgroup-level group scratch slots occupied at
epoch close when group scratch is supplied. If group scratch is absent, full, or
too collision-heavy, summary construction falls back to the coalescing-access
scan path; this does not by itself increment `coalescing_fallback_count`,
because the accesses were still recorded in the coalescing access log.

## Representation

Subgroup-level summaries use
`subgroup_level_context::coalesced_access_record`. Conceptually, a summary says:

> In this epoch, a contiguous range of lanes in one subgroup all performed the
> same kind of access, from the same static site, with the same byte size, and
> their addresses followed a fixed stride.

The fields are:

* `first_address`: address used by the first participating lane.
* `byte_count`: byte size of each individual access.
* `span_byte_count`: byte span covered from the lowest address to the end of
  the highest-addressed access.
* `first_lane`: first participating lane.
* `subgroup_id`: subgroup containing the participating lanes.
* `epoch`: shadow epoch where the accesses occurred.
* `kind`: load or store.
* `participant_count`: number of participating lanes.
* `valid`: whether the summary slot is populated.
* `address_stride`: signed byte stride between consecutive participating lanes'
  addresses.
* `site_id`: nonzero source-site id shared by the coalescing access records.

This representation can describe more than a strictly contiguous byte range.
For example, it can represent all of these shapes:

```text
lane 0: lds +  0 bytes
lane 1: lds +  4 bytes
lane 2: lds +  8 bytes
lane 3: lds + 12 bytes
```

```text
lane 0: lds +  0 bytes
lane 1: lds +  8 bytes
lane 2: lds + 16 bytes
lane 3: lds + 24 bytes
```

```text
lane 0: lds + 24 bytes
lane 1: lds + 16 bytes
lane 2: lds +  8 bytes
lane 3: lds +  0 bytes
```

The first has stride `4`, the second has stride `8`, and the third has stride
`-8`. A summary is only emitted when the stride is large enough that adjacent
participating accesses do not overlap.

## Conflict Detection

Thread-level conflict detection is exact-record based and does not use
coalescing summaries.

Subgroup-level conflict detection has two layers:

* on every default-site access, hip-moi appends an exact access record; for
  opted-in sites with coalescing access storage, it appends a lane-carrying
  `coalescing_access_record` instead, falling back to exact access records if
  that storage is absent or full;
* at `ctx.syncthreads()` or a final uniform `ctx.finish()`, summaries produced
  for the closing epoch are compared with each other and with exact or
  coalescing access records that were not themselves summarized;
* unsummarized exact and coalescing access records are compared with each other
  as the fallback path.

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

It only runs in subgroup-level mode at `ctx.syncthreads()` or a final uniform
`ctx.finish()`. Subgroup-level mode proves patterns from the separate
coalescing access log and can emit a summary when all of the following are true
for a group of records:

* the records have a nonzero `site_id`,
* they are in the same epoch,
* they are in the same subgroup,
* they have the same access kind,
* they have the same `byte_count`,
* their subgroup lanes form one contiguous range,
* each participating lane has exactly one record for that site in that epoch,
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
* subgroup-level opted-in accesses when no coalescing access storage was
  supplied; these accesses fall back to ordinary exact records,
* accesses that are not visible to hip-moi because they were raw LDS loads or
  stores instead of `ctx.lds_load` / `ctx.lds_store`.

The accelerated subgroup-level grouped path is not used when no group scratch
storage was supplied or when that scratch storage fills up; those cases fall
back to the older coalescing-access scan path.

This means coalescing is currently most useful as a way to observe and diagnose
simple, regular all-lane LDS access sites in subgroup-level mode. It has started
reducing exact access-log traffic for opted-in subgroup-level sites, but it is
still a conservative optimization path that users should validate on their
kernels.
