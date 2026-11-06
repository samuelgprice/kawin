"""Bitwise default-phase gates and frozen-fixture integrity, independent of engine sampling."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest


def runner():
    """Load the acceptance tool without importing research or thermodynamic code."""
    spec=importlib.util.spec_from_file_location('default_acceptance_runner',Path(__file__).parents[1]/'tools/acceptance.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def comparison_fixture(tmp_path, monkeypatch, actual):
    """Use tiny arrays solely to test report decisions, never to replace real generation."""
    tool=runner()
    monkeypatch.setattr(tool,'OUTPUT',tmp_path)
    monkeypatch.setattr(tool,'MANIFEST',tmp_path/'manifest.json')
    for owner in ['reference','package']:
        (tmp_path/owner).mkdir()
    report=dict(status='PASS',stop_status='default_phase_sampling_completed',
        default_phase_contract=dict(endpoint_phases=[['BCC_A2','FCC_A1'],['FCC_A1']]))
    wanted=np.array([0.]) if actual[0] == 0 else np.array([1.])
    key='provider/interfaceData/alpha'
    for case in ['O08','C01']:
        (tmp_path/'reference'/f'{case}.json').write_text(json.dumps(report))
        np.savez(tmp_path/'reference'/f'{case}.npz',**{key:wanted})
        np.savez(tmp_path/'package'/f'{case}.npz',**{key:actual})
    hashes={name:hashlib.sha256((tmp_path/'reference'/name).read_bytes()).hexdigest() for name in ['O08.json','O08.npz']}
    manifest=dict(default_phase_sampling=dict(reference_sha256=hashes),
        public_tolerances=dict(interface=dict(rtol=1e-10,atol=1e-12)))
    tool.MANIFEST.write_text(json.dumps(manifest))
    return tool,report,key


@pytest.mark.parametrize('actual',[np.array([-0.]),np.array([1.+1e-14]),np.array([1.],dtype=np.float32)])
def test_default_phase_requires_bytes_even_in_public_mode(tmp_path,monkeypatch,actual):
    """Signed zeros, small errors and dtype changes fail O08 despite passing old public gates."""
    tool,report,key=comparison_fixture(tmp_path,monkeypatch,actual)
    assert not tool.compare('C01',report,tolerance='public').get('failed_keys')
    result=tool.compare('O08',report,tolerance='public')
    assert result['failed_keys']==[key]
    metrics=json.loads((tmp_path/'package/O08.deviations.json').read_text())
    applied=metrics['quantities']['interface']['arrays'][key]['tolerance']
    assert applied['rtol']==0 and applied['atol']==0


def test_ordered_phase_metadata_is_part_of_default_phase_gate(tmp_path,monkeypatch):
    """Numerically equal endpoints cannot hide phase-label reversal."""
    tool,report,key=comparison_fixture(tmp_path,monkeypatch,np.array([1.]))
    report['default_phase_contract']['endpoint_phases'][0].reverse()
    with pytest.raises(AssertionError,match='labels/validity'):
        tool.compare('O08',report,tolerance='public')


def test_frozen_default_phase_metadata_hash_is_verified(tmp_path,monkeypatch):
    """Even a semantically harmless rewrite cannot silently replace a frozen fixture."""
    tool,report,key=comparison_fixture(tmp_path,monkeypatch,np.array([1.]))
    with (tmp_path/'reference/O08.json').open('a') as handle:
        handle.write(' ')
    with pytest.raises(AssertionError,match='frozen fixture changed'):
        tool.compare('O08',report,tolerance='public')


def test_unfrozen_default_phase_reference_cannot_pass(tmp_path,monkeypatch):
    """O08 must have its own recorded reference hashes before a gate can accept it."""
    tool,report,key=comparison_fixture(tmp_path,monkeypatch,np.array([1.]))
    manifest=json.loads(tool.MANIFEST.read_text())
    manifest['default_phase_sampling']['reference_sha256']={}
    tool.MANIFEST.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError,match='frozen fixture hashes'):
        tool.compare('O08',report,tolerance='public')


def test_reference_worker_refuses_to_overwrite_frozen_default_phase_fixture(tmp_path,monkeypatch):
    """Diagnostic reference runs require another destination; existing golden bytes stay intact."""
    tool=runner()
    monkeypatch.setattr(tool,'OUTPUT',tmp_path)
    path=tmp_path/'reference/O08.npz'
    path.parent.mkdir()
    path.write_bytes(b'frozen test fixture')
    with pytest.raises(AssertionError,match='refusing overwrite'):
        tool.worker('O08','reference')
    assert path.read_bytes()==b'frozen test fixture'
