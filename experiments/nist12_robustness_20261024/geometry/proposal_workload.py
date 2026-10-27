"""Label-free observation-size audit for planning CPU memory and runtime."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, read, save, ACTIVE_INPUTS

rows = []
active_scenes = {scene['scene_id'] for _, manifest, _, _ in ACTIVE_INPUTS
                 for scene in read(HERE/manifest)['scenes']}
for directory in sorted((HERE/'geometry/localization').iterdir()):
    if directory.name not in active_scenes:
        continue
    file = directory/'localization.json'
    if not file.exists():
        continue
    loc = read(file)
    with np.load(directory/'masks.npz') as archive:
        masks = archive['masks']
        sizes = np.count_nonzero(masks, axis=(1, 2)).tolist() if len(masks) else []
    rows.append({'scene_id': loc['scene_id'], 'proposals': len(sizes),
                 'observation_pixels': sizes, 'max_observation_pixels': max(sizes, default=0)})
save(HERE/'geometry/proposal_workload.json', {'scenes': rows, 'labels_read': False,
     'note': 'Frozen proposal-mask foreground sizes; no pruning or changed preprocessing.'})
for row in sorted(rows, key=lambda x:x['max_observation_pixels'], reverse=True)[:15]:
    print(row)
