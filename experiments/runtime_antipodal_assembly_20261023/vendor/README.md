# Vendored runtime antipodal contact generation

This experiment uses the existing Apache-2.0 EmbodiChain GraspKit sampler.
It does not install the full EmbodiChain framework or download model weights.

```python
import torch
from experiments.runtime_antipodal_assembly_20261023.vendor import (
    AntipodalSampler, AntipodalSamplerCfg,
)

sampler = AntipodalSampler(AntipodalSamplerCfg(
    n_sample=2048, max_angle=0.2617993877991494,
    min_length=0.001, max_length=0.08,
))
pairs = sampler.sample(
    torch.as_tensor(vertices, dtype=torch.float32),
    torch.as_tensor(faces, dtype=torch.int64),
)
```

Inputs: mesh vertices in metres, triangle indices. Output: an `[N, 2, 3]`
float32 tensor of opposing contact pairs in the input mesh's coordinate frame.
Pair index 0 is the opposite ray-hit contact; index 1 is the surface sample.
Passing mesh vertices already transformed into the workspace frame makes
sampling scene-conditioned without any stored part-specific grasp poses.
Sampling only creates contacts. The host must construct gripper frames, use the
actual gripper geometry, and check reachability, the target, table, approach,
closure, lift and any requested assembly path.

Runtime dependencies are NumPy, Open3D and PyTorch, plus Python's standard
library. They are already expected in the project's local experiment runtime.
No framework-owned simulator, robot assets, Viser frontend or caches are used.

The pinned upstream main version uses Fibonacci-sphere surface ray casting.
Its `max_angle` serves as the normal acceptance threshold; set it positive.
Its helper for random ray perturbation is not called by the pinned sample path.
This is a geometric candidate generator, not a force-closure proof or a learned
grasp-quality predictor. Finite contact sampling can miss feasible grasps.
See NOTICE.txt and provenance.json for the source and one import adaptation.

