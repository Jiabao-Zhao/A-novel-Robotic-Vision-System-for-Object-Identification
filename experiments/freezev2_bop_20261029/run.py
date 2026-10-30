"""Run the isolated reproduction from this directory; never imports production."""
from pathlib import Path
import argparse
import json
import sys

from freezev2_repro.data import write_json
from freezev2_repro.pipeline import load_config, local_path, doctor, geometry_onboard, feature_onboard, infer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['doctor', 'render', 'onboard', 'infer-one', 'infer', 'synthetic'])
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--output-root', help='Explicit separate cache/run directory; never overwrites old runs')
    parser.add_argument('--allow-oracle', action='store_true', help='Separate labeled oracle-mask diagnostic only')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_root:
        config['paths']['output_root'] = args.output_root
    output = local_path(config['paths']['output_root'])
    try:
        if args.command == 'doctor':
            result = doctor(config)
            write_json(output / 'environment_inventory.json', result)
        elif args.command == 'render':
            result = geometry_onboard(config)
        elif args.command == 'onboard':
            result = feature_onboard(config)
        elif args.command in ('infer-one', 'infer'):
            result = infer(config, single_mask=args.command == 'infer-one', allow_oracle=args.allow_oracle)
        else:
            from freezev2_repro.synthetic import run_synthetic
            result = run_synthetic(config)
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        result = {'status': 'blocked_or_failed', 'command': args.command, 'type': type(exc).__name__, 'reason': str(exc),
                  'claim': 'No benchmark or successful inference claim.'}
        write_json(output / f'{args.command}_failure.json', result)
        print(json.dumps(result, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
