"""Read-only computation status; never opens evaluation annotations."""
from common import HERE, read, ACTIVE_INPUTS, ACTIVE_SCENES, ACTIVE_QUERIES


def main():
    print(f'Current NIST scope: {ACTIVE_SCENES} inputs / {ACTIVE_QUERIES:,} CAD queries; synthetic depth corruption and T-LESS retired.')
    if (HERE/'finalization/completion.json').exists():
        print('FINALIZATION_COMPLETE: predictions locked, evaluation/report ready for visual review.')
        return
    for receipt in (HERE/'finalization').glob('*.json'):
        value = read(receipt)
        if value.get('returncode', 0):
            print('FINALIZATION_ERROR: '+receipt.name)
            return
    completed = sum((HERE/'baselines/nemo'/f'R{i:02d}'/'completion.json').exists()
                    for i in range(20))
    scenes = [scene for _, manifest, _, _ in ACTIVE_INPUTS
              for scene in read(HERE/manifest)['scenes']]
    localized = sum((HERE/'geometry/localization'/s['scene_id']/'localization.json').exists()
                    for s in scenes)
    index = HERE/'geometry/nist_pair_index_20261027.json'
    if index.exists():
        keys = {row['fit_key'] for rows in read(index)['scene_pairs'].values() for row in rows}
        fits = sum((HERE/'geometry/fits'/key[:2]/(key+'.json')).exists() for key in keys)
        print(f'NIST geometry cache: {fits:,}/{len(keys):,} saved summaries; '
              f'localization: {localized}/{ACTIVE_SCENES} retained inputs.')
    else:
        print(f'Localization: {localized}/{ACTIVE_SCENES} retained inputs; NIST cache index pending.')
    if (HERE/'visual/queue_completion.json').exists():
        print('CLIP: cached cohort outputs available. Native baselines: all six methods complete.'
              if (HERE/'baselines/gpu_release.json').exists()
              else f'CLIP complete; NeMO {completed}/20 scenes.')
        return
    progress = HERE/'visual/progress.json'
    if progress.exists():
        row = read(progress)
        print(f'Qwen descriptions {row["processed_descriptions"]}/{row["expected_descriptions"]}; '
              f'{row["reused"]} exact-image cache reuses.')


if __name__ == '__main__':
    main()
