import json,os,subprocess,sys
from pathlib import Path
from tabular_spectral_kernel.cli import RECORDED

def test_packaged_artifacts_available():
    assert (RECORDED/'manifest.json').is_file()
    assert (RECORDED/'spectral_stage1_extended/adult/seed0/standard/checkpoint.pt').is_file()

def test_cli_works_outside_checkout_with_spaces(tmp_path):
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    output=tmp_path/'output with spaces'
    for command in ['smoke','verify']:
        subprocess.run([sys.executable,'-m','tabular_spectral_kernel.cli',command,'--device','cpu','--output',str(output/command)],cwd=tmp_path,env=env,check=True,capture_output=True,text=True)
    assert json.loads((output/'verify/verification.json').read_text())['prediction_sets']==36
    assert json.loads((output/'smoke/smoke.json').read_text())['ridge_residual']<1e-8

def test_download_rejects_modified_cached_inputs(tmp_path):
    from tabular_spectral_kernel.datasets import prepare
    import pytest
    for name in ['adult.data','adult.test']:(tmp_path/name).write_text('modified')
    with pytest.raises(ValueError,match='differs from the paper input'):prepare('adult',tmp_path)
