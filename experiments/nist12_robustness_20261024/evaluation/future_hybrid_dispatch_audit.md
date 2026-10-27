# Future CPU plus one-GPU dispatch: source-only audit

Status: feasible as a resource-only follow-up, conditional on the authorized dense GPU pilot succeeding and actual host capacity. **No implementation or production integration is approved by this note.** The sole current GPU fit and every source/input it pins must remain unchanged through commit and verified exit. This audit read sources and existing resource receipts only; it ran no query, fit, model or accuracy evaluation.

## Small dispatch change

`geometry/run_depth.py:204-238` already partitions unique missing keys into ordinary, dense and very-dense stages, but submits every job to one CPU executor. Keep the existing ordinary stage unchanged. For a later dense stage, replace only the submit-all/completion loop with a common iterator feeding four CPU slots and one sequential GPU slot. For very-dense work, two CPU slots plus one GPU slot is a **maximum proposal, not an established safe capacity**.

The following is a 28-line scheduling sketch, not implemented code. The two spawn executors, stage admission, reviewed GPU helper and unchanged completion bookkeeping must exist outside this block. `cpu_slots` is the admitted CPU process count, not the number of query threads. `gpu_fit_job` must ultimately call the unchanged original `fit_job`.

```python
assert len({job[3] for job in group}) == len(group)
todo, exhausted = iter(group), False
pending, owners = {}, {}
free = [cpu_pool] * cpu_slots + [gpu_pool]
while pending or not exhausted:
    while free and not exhausted:
        pool = free.pop()
        try:
            job = next(todo)
        except StopIteration:
            exhausted = True
            free.append(pool)
            break
        key = job[3]
        assert key not in owners
        call = gpu_fit_job if pool is gpu_pool else fit_job
        future = pool.submit(call, job)
        pending[future], owners[key] = (key, pool), future
    if not pending:
        break
    ready, _ = wait(pending, return_when=FIRST_COMPLETED)
    for future in ready:
        key, pool = pending.pop(future)
        assert future.result() == key
        verify_commit_and_record_existing_progress(key, pool)
        del owners[key]
        free.append(pool)
assert exhausted and not pending and not owners
```

Each iterator entry is consumed once. There are at most five outstanding futures, and at most one belongs to the one-worker GPU executor. Whichever backend finishes first receives the next unassigned key. There is no separate CPU copy of the GPU queue. Existing per-scene key sets and `scene_scores` updates are used after successful results, followed by the same final every-scene check.

This proves one owner **within the single cohort dispatcher**. It is not an operating-system-wide lock: preserve the serial cohort queue and the pre-index pilot/reservation handoff. Do not launch auxiliary writers during a stage or release a failed key while its former worker can still write. The existing missing-key scan alone cannot protect against another independent process starting the same key.

## GPU worker and failure contract

Use spawn, never fork an initialized CUDA context. Give the GPU executor one process and at most one outstanding job. A minimal first version should create and close a runtime **per job**, retain the cleanup-before-original-summary-commit rule, and restore the original bare CAD tree after wrapping. A cached CAD reused by the next job must not retain a closed proxy. This avoids changing the reviewed pilot's context lifetime; its measured context/module startup is small compared with minute-scale dense fits.

A persistent CUDA context is possible but is a larger review: current `profile_required_gpu_pin.py:229-263` explicitly closes it before scientific commit. Replacing that with per-job synchronization/error/FP checks and worker-final teardown requires a new commit/lifecycle proof, not merely the dispatch loop above. Per-job GPU/certified/fallback statistics must be fresh counters or deltas; an earlier job's positive GPU count must not make an all-CPU later job pass. Preserve original points, tree order/nodes, native sampling and all pose hypotheses; initialize GPU workers through the plain initializer and bare-tree guard, avoiding the CPU query proxy.

Keep the full-fit, source/input, ownership, lock, resource and clean-teardown guards before the unchanged original writer. A GPU error or disabled guard must fail its future without manufacturing an unavailable score. The simplest initial policy is to stop further dispatch and fail that cohort, allow already-running valid CPU jobs to finish, close/verify the GPU worker, preserve logs and orphan archive hashes, and leave missing keys pending for an explicitly authorized resume. Do not immediately submit the same key to CPU while the GPU worker might still commit it. Genuine native all-projection-unavailable results retain their original full-bank status.

The scheduling sketch is small; a safely generalized GPU job helper, admission logic and evidence records are additional work. It would be misleading to describe the whole integration as a 28-line patch.

## Host and device capacity

Known evidence supplied by geometry, without new probes:

- Completed CPU pin at1,279,564 voxels: peak RSS962,328KiB, about939.8MiB.
- CPU Gear at1,358,622 voxels: a21:23 snapshot showed PSS605.4MiB/RSS765.2MiB; this snapshot is not its lifetime peak.
- Current GPU pilot at1,358,622 voxels: a21:23 snapshot showed PSS746.2MiB/RSS906.8MiB. Its1.5GiB **incremental** cap covers preparation, fitting, dimensions, compression and archive creation through the pre-summary high-water check. That guard detects a peak; it is not an allocation limiter. Its completed peak was not available when this audit began.
- No completed fit above2,000,000 voxels establishes CPU or GPU capacity yet. The noise5mm example contains3,340,699 voxels and4,488,713 raw foreground pixels. Full host observation/refinement/size arrays grow with those counts even though device query chunks remain262,144 rows.

Do not carry the1.5GiB pilot cap into the greater-than2M stage as if validated. It may be insufficient. Use a separately authorized already-required large job to measure the whole worker peak, then prospectively set and review that stage's cap. Neither two CPU workers nor a GPU worker is automatically admitted merely because its count appears in the current schedule.

Before warming a stage, require actual physical host headroom for **all incremental resident workers plus at least2GiB reserve**, including any still-running pilot, parent indexing/preparation, native raycasters and other applications. If workers are already warm, do not count their resident memory twice; allow for their remaining peak growth. Check actual Windows physical availability and WSL availability, not summed shared RSS alone. Lower admitted CPU concurrency or postpone GPU dispatch when the reserve cannot be retained. No user applications or global environments are changed.

The nominal dense plan uses CPU4 processes ×4 query threads plus GPU fallback4, up to20 query threads when fallback is active. The very-dense proposal uses CPU2 ×8 plus GPU fallback4, also20. These counts exclude any remaining auxiliary job; record and admit those separately. Aggregate jobs completed per wall time and host reserve matter more than isolated query speed.

The128MiB explicit-device limit remains unchanged; there must be only one CUDA context owner executing fits. The current CAD snapshot cache is4MiB per CAD,16MiB total and32 entries. Twelve CADs with at most40,000 sampled points each fit the snapshot budget, but native `_CAD` meshes/raycasters and `_OBS` arrays are separate host allocations. A larger catalog can exceed the snapshot budget. Per-job runtimes avoid cumulative device/snapshot-cache growth, but a persistent Python worker still needs a lifetime/stage host cap; resetting its baseline for every job would hide retained growth. Any cache eviction or worker recycling needs its own exact-reuse review. `native.sample_mesh` is deterministic seed0, but that fact alone is not a complete raycaster/cache-lifecycle validation.

## Receipts, finalization and approval gates

After the running pilot's verified exit and an explicit root integration decision, preserve the old runner and write a prospective resource adapter receipt. Keep `observation`, `observation_key`, `fit_job`, `scene_scores`, `batched.py`, `dimensions.py`, native geometry and execution-key formula identical; compare their source/AST hashes. The loop changes only which process owns a complete unchanged job. No partial banks, decision-only shortcuts, pose reductions or score substitutions are allowed.

Update the future execution/progress/completion resource fields honestly: CPU/GPU process counts, each backend's query threads, GPU threshold, query chunk size, host/device caps, admitted reserve, backend per completed key, source/proof/build hashes and teardown results. Normal cache summaries/banks keep their existing schema; backend evidence belongs in sidecar receipts. Preserve the serial queue's per-cohort launch/source snapshots, attempt archives and return-code checks. A failed stage cannot write its completion marker.

Do not publish geometry `queue_completion.json` until both executors are drained, every worker can no longer write, GPU teardown is verified, every scene's existing score completeness check passes and all required sidecar evidence is finalized. Then the existing finalizer may verify and lock predictions before accuracy access. The evaluator already checks every referenced bank hash and full query set, but its present inventory does **not automatically enumerate new geometry resource JSON sidecars**. Any such evidence must enter a separately approved verified inventory/anchor before lock; do not claim it is locked solely because it exists. No frozen evaluator/protocol edits are made in this audit.

Required future gates: successful complete required GPU pilot; actual representative capacity; reviewed generic worker/commit/failure helper; focused synthetic one-owner/error/resume tests and exact reused-bank parity; prospective resource provenance; explicit root authorization. Current source pins prohibit integration until pilot commit and exit. The user's full-scores-versus-decision-only preference remains unanswered; all original full searches continue. Exact nearest-neighbor query certificates preserve full banks and are a different mechanism.

Source references: `geometry/run_depth.py:62-142,161-267`; `geometry/batched.py:7-8,68-118`; `geometry/pilot_handoff.py:13-78`; `geometry/queue_cohorts.py:77-127`; `finalize_study.py:9-40`; `evaluation/evaluate.py:123-241`; `evaluation/fp32_certificate_audit/certified_query_proxy.py:32-38,81-225,239-342`; `profile_required_gpu_pin.py:212-286`; original native `geometry.py:36-49,260-268`. This new note is the only file created by this task.
