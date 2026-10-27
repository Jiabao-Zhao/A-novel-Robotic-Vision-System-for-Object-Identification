# Controlled complete-proxy query benchmark

The authorized query-only matrix passed. All 15 uninstrumented CPU4/proxy pairs and both separately instrumented diagnostic calls returned identical distance/index bytes. All work used the same 10,651-point Gear CAD; no fit, new scientific key, model or accuracy annotation was used. Production sources and scientific caches were unchanged.

The primary table retains all three repetitions. Ratio means total CPU4 duration divided by total proxy duration, including conversion, transfers, safeguards, certification, original-SciPy fallback and selected-distance recomputation.

| Query workload | CPU4 median (ms) | Proxy median (ms) | Three-pair aggregate ratio |
|---|---:|---:|---:|
| Dense 16,384 rows | 9.960 | 24.755 | 0.456x |
| Dense 65,536 rows | 35.707 | 16.999 | 0.664x |
| Dense 131,072 rows | 67.646 | 31.792 | 2.106x |
| Dense 262,144 rows | 124.300 | 66.355 | 1.840x |
| Nominal 127,530 rows | 33.274 | 25.429 | 1.300x |

The 65,536-row proxy durations were **126.415, 16.672 and 16.999 ms**. The outlier is retained and its cause is unmeasured. The favorable median does not establish an aggregate win at this size. At 16,384 rows the proxy lost in all three pairs. The 131,072- and 262,144-row dense pairs consistently won; the nominal pairs won by a smaller margin. Across the complete mixed matrix, CPU4 took 0.820572 seconds and the proxy 0.615589 seconds, a 1.333x aggregate ratio.

The nominal workload is the fixed 4,905-point observation transformed using the first 26 cached predicted poses. It is not every intermediate query from the full-bank fit. The dense workload is the previously prepared, hash-pinned 262,144-row bank and deterministic nested subsets. Dense certification was about 90.94%; nominal certification was 98.37%. Ratios do not establish speed for another CAD size or whole fits.

## Diagnostic overhead

Observers were installed only after all primary pairs. The following are two single-call diagnostic measurements, not medians. Parent timer durations are not double-counted with nested children.

| Component | Dense 262k (ms) | Nominal 127,530 (ms) |
|---|---:|---:|
| Complete instrumented proxy | 63.968 | 25.144 |
| Tree/data/order/opaque-node fingerprint | 0.330 | 0.322 |
| FP and resource guards | 0.081 | 0.073 |
| Cached CAD equality check | 0.035 | 0.028 |
| Per-row bounds | 14.032 | 6.603 |
| Kernel launch plus synchronization | 20.159 | 9.704 |
| Host-to-device plus device-to-host transfer | 1.476 | 0.593 |
| Other exclusive `top_two` work | 1.759 | 0.666 |
| CPU selected-distance recomputation | 3.683 | 1.263 |
| Unassigned remainder | 22.412 | 5.893 |

The remainder includes original-SciPy fallback, certificate masks/gathers, output placement, other Python work and observer bookkeeping. These were not separately measured. Tree fingerprints, FP checks and cached-CAD checks together took about 0.446 ms and 0.422 ms. Per-row bounds cost substantially more. Source/review/binary-file hashes are setup work, not a per-query file-hashing loop. The measured empty observer overhead was about 0.861 microseconds per call.

## Window, resources and cleanup

All input preparation, context initialization and five workload warmups finished before READY. The uninstrumented matrix plus two diagnostics took 1.533329 seconds. Geometry independently verified and paused eight scientific workers and one Gear pilot, then resumed all nine after **1.735949 seconds**. Every PID/start-tick/full-argv identity matched; post-resume checks found none stopped or zombie. The independent watchdog also completed its early-release resumes. No production restart occurred.

READY preceded GO by about 52.49 seconds while geometry sampled resources and armed the guard. The initialized context remained idle during that interval. GPU clocks and scheduling were not recorded; no causal attribution of the observed latency outliers to this interval is established. Three repetitions and a short window do not measure stable tail latency.

Fresh available host memory was 6.1359 GiB before STOP and 6.0866 GiB after CONT. Peak benchmark RSS was 266.97 MiB total / 200.04 MiB incremental. Explicit device peak was 7.122 MiB. Cleanup errors and release errors were empty; the context was destroyed and retained tracked device bytes were zero. Windows host CPU/memory and WSL load snapshots are saved. Protected vmmem CPU counters were unavailable, so no numerical separation of unrelated host CPU load is claimed. WSL load averages include preceding scientific work and are not instantaneous controlled-window utilization.

## Evidence and decision

Run: `controlled_queries/20261024T204249442871Z/`.

- Completion SHA-256: `a57726acc3fc2a3186cf2b8d568d8b26ba4c9abba4ffbe3fb4bbbff7ad2c954d`.
- Runner SHA-256: `8da9254bc7891bfdfb67cd2a583ee800b7303e995c5afbf999fae7ae7b14f3fb`.
- V2 independent source-review SHA-256: `1916e2813343d7ea0b7ba7af9c044b34f81fd8af76a81c43aa27526ff827fa66`.
- Guarded controller SHA-256: `d99e342fe0d30b7a3acc1aa64c2aba3d08ba04a1ee285e13c3868b608b6a53bf`.
- Geometry participant/resource/resume evidence: `../../geometry/gpu_proxy_cpu_windows/20261024T204249442871Z/`.

This supports further consideration for large query batches on this CAD. It does **not** support the current 16,384-row activation threshold as a general throughput choice. No threshold or live runner was changed. The earlier 432-hypothesis replay established whole-bank exactness; a matched controlled CPU4/GPU bank pair would still be needed to claim whole-fit acceleration. The audit's 1 GiB increment ceiling remains an audit budget, not a production preprocessing allowance.
