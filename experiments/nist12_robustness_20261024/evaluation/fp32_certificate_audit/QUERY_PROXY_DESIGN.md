# Isolated cached-bank query proxy

`certified_query_proxy.py` is a new technical execution adapter. No scientific
runner imports it. The proposed replay replaces only one fresh local CAD
object's `.tree` field, never changes a source kernel, point sample, support/yaw
hypothesis, score or cache key, and writes only to a new technical audit folder.

One owner CPU thread manages one lazy NVRTC module/context and shared bounded
query/output buffers. Original CAD point order is retained in separate cached
FP32 device buffers; a CAD switch never overwrites another CAD buffer. CPU
snapshots are limited to4MiB per CAD,16MiB total including4KiB metadata per
entry, and32 entries. Budgets are checked before allocations/snapshots; exceeding
them disables the isolated GPU path and forwards the original CPU call. Explicit
device buffers remain capped128MiB and process RSS increment1GiB. The runtime
must be closed on its owner thread; partial initialization/error cleanup is
explicit, idempotent, and saved.

Only a bare original SciPy cKDTree can be wrapped. Periodic/non-3D/empty/one-point/
oversized trees fall back without copying their data. Supported trees retain an
original data reference; the runtime makes only budgeted read-only snapshots.
Before acceleration each query checks a fingerprint of original data, indices,
bounds/metadata and opaque internal node/split state. Changed/stale trees stop
accelerating. Concurrent mutation of original tree/input data is unsupported,
as it is for the original SciPy tree. Owner-thread checking prevents concurrent
CUDA buffer use; foreign-thread calls use the original CPU query.

Supported accelerated calls have keyword/default scalar k=1,p=2,eps=0,infinite
distance bound and a valid workers setting, on finite NumPy FP32/FP64 arrays with
final dimension3. Positional/other/unknown options, unsupported dtype/shape,
nonfinite input, and fewer than16,384 eligible rows use the original tree with
exact original args/kwargs, including its exceptions and output shape/types,
except the explicitly supported small-query resource schedule described below.
The threshold is the smallest directly measured successful size; smaller tail
chunks also use CPU. Supported calls flatten/restore original batch dimensions
and process consecutive chunks of at most262,144 rows, preserving order.

Every chunk uses the reviewed per-row positive quadratic, supported-domain mask,
and all raw finite/nonnegative/order/index/invalid flags before certification.
Only eligible finite/capped rows enter the kernel. Uncertified rows and supported
small calls use the same original tree with runtime.cpu_workers=4 for this
replay, explicitly matching the prior validated four-thread execution schedule.
The controlled runtime setting also supports1/2/8 for future resource-only
trials; none is being installed into production. Unsupported whole calls preserve
the caller's worker argument, including overflow/invalid-worker exceptions.
Distances for certified indices use the pinned CPU operation sequence
and hardware square root. All original/new math/source/binary/FP/PTX guards from
the isolated pilots remain enforced. CUDA execution/guard failures discard any
partial chunk result and forward the entire original CPU call, with the cause
recorded; a replay cannot pass if the runtime disabled itself.

`test_certified_query_proxy.py` uses a clearly fake CPU-FP32 backend and does not
initialize CUDA. It tests exact fallback/exception forwarding, scalar/empty/
strided/multibatch output, raw failures, ties, stale inputs, foreign threads,
bounded caches, partial-init cleanup and chunk/tail order.

`replay_cached_bank.py` is gated on saved independent wrapper-review hashes and
a fresh geometry-agent memory clearance. It accepts exactly one fixed cached
case: R00/Gear_Large/object1/proposal0,4905observed voxels,
key2ae9e963e1f146d8a54d1b735d4323d241066513fd4381b8b74aa23d2fe7628c,
bankSHA9f5e8f79473d992d6d3ceb75cf93e5167d48dabdaf52ef2118bb7bc8f550514b.
All432hypotheses must compare exactly: every non-timing key, ordered list, type,
value, float64 bit pattern and canonical JSON byte. Only `seconds` fields are
ignored. Scientific dependencies/cache files must remain byte-identical, GPU
calls/certified rows must be positive, runtime errors absent, and cleanup must
report no errors and zero retained device bytes. It records full replay wall
time; historical cached time alone is not a paired speed control.

No uncached required key, new accuracy annotation, model or production
installation is authorized by this design. Wrapper review and immediate resource
confirmation are still required before the prospective replay executes.
