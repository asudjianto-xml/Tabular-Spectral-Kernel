"""Portable paths and integrity checks for exported research workspaces."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent


def device():
    import torch
    requested = os.environ.get('TSK_DEVICE', 'auto')
    if requested not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('TSK_DEVICE must be auto, cpu or cuda')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu')
    return ('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested


def bike_path():
    explicit = os.environ.get('TSK_BIKE_CSV')
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(os.environ.get('TSK_DATA_DIR', str(ROOT / 'data'))).expanduser().resolve() / 'hour.csv'


def require_bike(path):
    from tabular_spectral_kernel.datasets import TEXT_HASHES
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Bike Sharing data are missing. Run tsk download --dataset bike --data-dir <cache>, then pass --data-dir <cache> to tsk-context run.')
    if hashlib.sha256(path.read_bytes()).hexdigest() != TEXT_HASHES['hour.csv']:
        raise ValueError('hour.csv does not match the verified UCI Bike Sharing input')


def output_path(script, default):
    override = os.environ.get('TSK_CONTEXT_OUTPUT')
    selected = os.environ.get('TSK_CONTEXT_SCRIPT')
    if override and selected and Path(script).resolve() == Path(selected).resolve():
        return Path(override).expanduser().resolve()
    return default


def verify_artifact(name, expected):
    """Validate original artifacts or declared portability transformations.

    The release manifest retains both original and distributed hashes. This
    does not claim the ported source is byte-identical to the training source.
    Fresh output manifests are checked directly against their stated digest.
    """
    from tabular_spectral_kernel._paper import identifiable_spectral_california as model
    if name == '@canonical_model':
        path = Path(model.__file__)
        key = name
    elif name == '@bike_data':
        path = bike_path()
        key = name
    else:
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        try:
            key = path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            key = None
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual == expected:
        return
    manifest = json.loads((ROOT / 'release_manifest.json').read_text())
    record = manifest.get(key, {})
    if actual != record.get('sha256') or expected != record.get('original_sha256'):
        raise ValueError(f'Artifact digest mismatch: {name}')
