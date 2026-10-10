import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from tabular_spectral_kernel.context_cli import BUNDLE, export_workspace, verify_bundle


def runtime():
    spec = importlib.util.spec_from_file_location('test_context_runtime', BUNDLE / 'context_runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recorded_artifacts_and_metrics():
    verify_bundle()


def test_export_is_separate_and_refuses_overwrite(tmp_path):
    destination = export_workspace(tmp_path / 'source only')
    assert (destination / 'synthetic_drift/run.py').is_file()
    assert not (destination / 'synthetic_drift/unemployment_results').exists()
    with pytest.raises(FileExistsError):
        export_workspace(destination)


def test_ported_hashes_are_declared_and_tampering_fails(tmp_path):
    r = runtime()
    manifest = json.loads((BUNDLE / 'release_manifest.json').read_text())
    key = 'synthetic_drift/run.py'
    r.verify_artifact(key, manifest[key]['original_sha256'])
    with pytest.raises(ValueError, match='digest mismatch'):
        r.verify_artifact(key, '0' * 64)
    copied = export_workspace(tmp_path / 'modified workspace')
    r.ROOT = copied
    (copied / key).write_text('print("changed")\n')
    with pytest.raises(ValueError, match='digest mismatch'):
        r.verify_artifact(key, manifest[key]['original_sha256'])


def test_missing_or_changed_data_fail_without_download(tmp_path, monkeypatch):
    r = runtime()
    monkeypatch.setenv('TSK_BIKE_CSV', str(tmp_path / 'custom data.csv'))
    assert r.bike_path() == tmp_path / 'custom data.csv'
    with pytest.raises(FileNotFoundError, match='tsk download'):
        r.require_bike(r.bike_path())
    r.bike_path().write_text('not the verified data')
    with pytest.raises(ValueError, match='does not match'):
        r.require_bike(r.bike_path())


def test_cpu_device_and_targeted_output_override(tmp_path, monkeypatch):
    r = runtime()
    monkeypatch.setenv('TSK_DEVICE', 'cpu')
    assert r.device() == 'cpu'
    target = tmp_path / 'run.py'; other = tmp_path / 'dependency.py'; default = tmp_path / 'default'
    monkeypatch.setenv('TSK_CONTEXT_SCRIPT', str(target))
    monkeypatch.setenv('TSK_CONTEXT_OUTPUT', str(tmp_path / 'new output'))
    assert r.output_path(target, default) == tmp_path / 'new output'
    assert r.output_path(other, default) == default


def test_installed_cli_runs_outside_checkout_with_spaces(tmp_path):
    workspace = export_workspace(tmp_path / 'exported workspace')
    env = dict(os.environ); env.pop('PYTHONPATH', None)
    subprocess.run([sys.executable, '-m', 'tabular_spectral_kernel.context_cli', 'run',
                    '--workspace', str(workspace), '--device', 'cpu',
                    'bikeshare_context_fusion/smoke.py', '--', '--output', 'smoke output'],
                   cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    result = json.loads((workspace / 'smoke output/smoke.json').read_text())
    assert result['device'] == 'cpu'
    assert result['training_losses'][-1] < result['training_losses'][0]
    assert result['minimum_gram_eigenvalue'] >= -1e-10


def test_bundle_has_no_machine_paths_or_manuscripts():
    for path in BUNDLE.rglob('*'):
        if not path.is_file():
            continue
        assert path.suffix not in {'.tex', '.pdf', '.bib', '.bbl'}
        if path.suffix in {'.py', '.json'}:
            text = path.read_text()
            assert '/home/' not in text
            assert '/Users/' not in text
