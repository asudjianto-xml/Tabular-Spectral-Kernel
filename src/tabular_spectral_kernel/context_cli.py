"""Export and run the temporal-context experiments from an installed package."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BUNDLE = Path(__file__).with_name('context_bundle')


def export_workspace(output, recorded=False):
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    for src in BUNDLE.rglob('*'):
        if not src.is_file() or '__pycache__' in src.parts:
            continue
        if not recorded and src.suffix != '.py' and src.name != 'release_manifest.json':
            continue
        dst = output / src.relative_to(BUNDLE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return output


def verify_bundle():
    from ._paper import identifiable_spectral_california as model
    manifest = json.loads((BUNDLE / 'release_manifest.json').read_text())
    for name, record in manifest.items():
        path = Path(model.__file__) if name == '@canonical_model' else BUNDLE / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Release digest mismatch: {name}')
    # Recompute the principal hourly errors from saved predictions.
    import numpy as np
    import pandas as pd
    rows = []
    for seed in range(3):
        directory = BUNDLE / 'review_experiments' / 'bike_results' / f'seed{seed}'
        predictions = pd.read_csv(directory / 'predictions.csv')
        metrics = json.loads((directory / 'metrics.json').read_text())
        for method, row in metrics.items():
            error = predictions[method].to_numpy() - predictions['target'].to_numpy()
            np.testing.assert_allclose(np.sqrt(np.mean(error**2)), row['rmse'], rtol=1e-8, atol=1e-6)
            np.testing.assert_allclose(np.mean(np.abs(error)), row['mae'], rtol=1e-8, atol=1e-6)
            rows.append(row)
    print(json.dumps({'files_verified': len(manifest), 'bike_metric_rows_verified': len(rows)}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export', help='Create a separate writable experiment workspace')
    export.add_argument('--output', required=True, type=Path)
    export.add_argument('--recorded', action='store_true', help='Include checkpoints, predictions and recorded metrics')
    sub.add_parser('verify', help='Verify bundled hashes and recorded hourly prediction errors')
    sub.add_parser('list', help='List the available experiment scripts')
    run = sub.add_parser('run', help='Run a script in an exported workspace')
    run.add_argument('--workspace', required=True, type=Path)
    run.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    run.add_argument('--data-dir', type=Path)
    run.add_argument('--bike-csv', type=Path)
    run.add_argument('--output', type=Path, help='Override the selected review script output directory')
    run.add_argument('script', help='Workspace-relative script, e.g. synthetic_drift/run_credit_card.py')
    run.add_argument('args', nargs=argparse.REMAINDER, help='Arguments for that script, following --')
    args = parser.parse_args(argv)
    if args.command == 'export':
        print(export_workspace(args.output, args.recorded))
    elif args.command == 'verify':
        verify_bundle()
    elif args.command == 'list':
        for path in sorted(BUNDLE.glob('*/*.py')):
            if not path.name.startswith('test_'):
                print(path.relative_to(BUNDLE).as_posix())
    else:
        workspace = args.workspace.expanduser().resolve()
        script = (workspace / args.script).resolve()
        if not script.is_relative_to(workspace) or not script.is_file() or script.suffix != '.py':
            parser.error('script must be an existing Python file inside the workspace')
        env = dict(os.environ, TSK_DEVICE=args.device, TSK_CONTEXT_SCRIPT=str(script))
        env.pop('TSK_CONTEXT_OUTPUT', None)
        if args.data_dir:
            env['TSK_DATA_DIR'] = str(args.data_dir.expanduser().resolve())
        if args.bike_csv:
            env['TSK_BIKE_CSV'] = str(args.bike_csv.expanduser().resolve())
        if args.output:
            if script.parent.name != 'review_experiments':
                parser.error('--output is for review scripts; use the synthetic/bike script own --output or --out after --')
            destination = args.output.expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            env['TSK_CONTEXT_OUTPUT'] = str(destination)
        forwarded = args.args[1:] if args.args[:1] == ['--'] else args.args
        result = subprocess.run([sys.executable, str(script), *forwarded], cwd=workspace, env=env)
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
