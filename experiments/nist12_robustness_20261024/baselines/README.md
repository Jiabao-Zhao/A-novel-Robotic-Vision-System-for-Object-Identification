# Twenty-layout full-scene baseline extension

`protocol.json` freezes the adapter, source runners, exact CAD-only reference
caches, and label-free inputs before new inference. `run_all.py` isolates each
GPU stage in a separate process. No scene crops, masks, labels, poses, camera
extrinsics, target count, or outputs from our localizer enter these methods.

The adapter imports the immutable 20261023 runners. It redirects their input and
output paths, changes only the old experiment-specific eight-scene assertions
and completion counts to twenty, and reuses their exact CAD reference tensors.
All segmentation, native preprocessing, matching, filters, class assignment,
and NMS remain in the pinned official implementations and existing execution
compatibility adapters.

CNOS and SAM-6D run their own SAM generators. SAM-6D RGB and RGB-D share the same
SAM-6D proposals because they are scoring variants of that method. NIDS uses
its own Grounded-SAM; NOCTIS uses its own Grounded-SAM2.1 route. NeMO runs its
released multiscale full-scene conditional segmentation independently per CAD.
The NeMO memories use the previously corrected 32-view custom-CAD Pyrender
references; this is not claimed to reproduce exact BOP PBR onboarding.

`nids_basic_finite_templates` is the practical **no-adapter** NIDS control.
Exactly the six previously identified nonfinite FFA template slots for USB
(zero-based object 5, slots 16, 19, 20, 21, 24, 25) are assigned negative infinity
before native max. Every finite similarity remains exactly unchanged. The
guard asserts this on each call and preserves all twelve CAD classes. This
adjustment must be named in tables; it is not the unmodified native NIDS route.
That route has the already-documented invalid-reference failure and is not
rerun here solely to produce another zero-detection result.

Per-scene outputs are `<method>/Rxx/detections.json`, `masks.npz`, and
`completion.json`. Native proposal banks and intermediate scores are retained.
No geometric fitting from our method is added to these baseline scores.
