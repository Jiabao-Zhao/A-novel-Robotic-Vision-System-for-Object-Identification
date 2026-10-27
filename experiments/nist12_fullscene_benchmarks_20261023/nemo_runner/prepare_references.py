"""Freeze CAD-only view selection before NeMO inference; verify rendered sources."""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import digest, read, relative, save


def main():
    indices = np.sort(np.random.default_rng(0).choice(42, size=32, replace=False)).tolist()
    selection = {'seed': 0, 'algorithm': 'numpy.default_rng.choice(42,32,replace=False), sorted',
                 'ordinal_indices': indices, 'scene_information_used': False}
    selection_file = HERE / 'view_selection.json'
    if selection_file.exists():
        assert read(selection_file) == selection
    else:
        save(selection_file, selection)
    template_root = HERE.parent / 'templates42'
    if not (template_root / 'completion.json').exists():
        print('View selection frozen; waiting for complete custom CAD-render bank.', flush=True)
        return
    metadata = read(template_root / 'metadata.json')
    canonical_indices = [metadata['index_templates'][index] for index in indices]
    manifest = read(HERE.parent / 'inputs.json')
    references = {
        'route': 'Official NeMO encoder and detector with custom 32 CAD renders; not exact BOP PBR reproduction.',
        'render_route': 'Official CNOS Pyrender 42 CAD-only views; prospectively fixed seed0 subset of32.',
        'view_selection_sha256': digest(selection_file),
        'metadata_sha256': digest(template_root / 'metadata.json'),
        'canonical_view_indices': canonical_indices,
        'objects': {},
    }
    for item in manifest['objects']:
        images = [template_root / f'obj_{item["object_id"]:06d}' / f'{index:06d}.png'
                  for index in canonical_indices]
        assert len(images) == 32 and all(filename.is_file() for filename in images)
        references['objects'][str(item['object_id'])] = {
            'target': item['target'], 'images': list(map(relative, images)),
            'sha256': {relative(filename): digest(filename) for filename in images},
        }
    destination = HERE / 'references.json'
    if destination.exists():
        assert read(destination) == references
    else:
        save(destination, references)
    print('Verified 32 custom CAD references per object for all12 objects.', flush=True)


if __name__ == '__main__':
    main()
