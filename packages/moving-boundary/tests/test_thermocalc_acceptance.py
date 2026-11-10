"""O06 byte gates, frozen reference integrity and the one allowed metadata difference.

Tiny arrays below test comparison machinery only; no simulated TC provider is
used for generation or for the real acceptance case.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest


def runner():
    """Load acceptance tooling without importing TC-Python or research modules."""
    spec = importlib.util.spec_from_file_location('tc_acceptance_test', Path(__file__).parents[1]/'tools/acceptance.py')
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def fixture(tmp_path, monkeypatch, actual):
    """Create only comparison-unit-test data, with immutable-reference hashes."""
    tool = runner()
    monkeypatch.setattr(tool, 'OUTPUT', tmp_path)
    monkeypatch.setattr(tool, 'MANIFEST', tmp_path/'manifest.json')
    for name in ['reference','package']:
        (tmp_path/name).mkdir()
    report = dict(status='PASS',stop_status='completed')
    key = 'interfaceData/values'
    np.savez(tmp_path/'reference/O06.npz', **{key:np.asarray([1.])})
    np.savez(tmp_path/'package/O06.npz', **{key:actual})
    (tmp_path/'reference/O06.json').write_text(json.dumps(report))
    hashes = {name:hashlib.sha256((tmp_path/'reference'/name).read_bytes()).hexdigest()
              for name in ['O06.json','O06.npz']}
    tool.MANIFEST.write_text(json.dumps(dict(thermocalc_sampling=dict(reference_sha256=hashes), contracts=[],
        public_tolerances=dict(interface=dict(rtol=1e-10,atol=1e-12), profiles=dict(rtol=1e-10,atol=1e-12)))))
    return tool, report, key


@pytest.mark.parametrize('actual', [np.asarray([1.+1e-14]),np.asarray([1.],dtype=np.float32)])
def test_tc_numerical_arrays_require_bytes_in_public_mode(tmp_path, monkeypatch, actual):
    """Old public numerical tolerance cannot hide O06 roundoff or dtype changes."""
    tool, report, key = fixture(tmp_path,monkeypatch,actual)
    assert tool.compare('O06',report,tolerance='public')['failed_keys'] == [key]
    measured = json.loads((tmp_path/'package/O06.deviations.json').read_text())
    assert measured['quantities']['interface']['arrays'][key]['tolerance']['rtol'] == 0
    assert measured['quantities']['interface']['arrays'][key]['tolerance']['atol'] == 0


def test_tc_signed_zero_requires_bytes(tmp_path, monkeypatch):
    """Exact numerical equality cannot hide signed-zero changes."""
    tool, report, key = fixture(tmp_path,monkeypatch,np.asarray([-0.]))
    np.savez(tmp_path/'reference/O06.npz', **{key:np.asarray([0.])})
    manifest = json.loads(tool.MANIFEST.read_text())
    manifest['thermocalc_sampling']['reference_sha256']['O06.npz'] = hashlib.sha256((tmp_path/'reference/O06.npz').read_bytes()).hexdigest()
    tool.MANIFEST.write_text(json.dumps(manifest))
    assert tool.compare('O06',report,tolerance='public')['failed_keys'] == [key]


@pytest.mark.parametrize('filename', ['O06.json','O06.npz'])
def test_tc_reference_hashes_are_checked(tmp_path, monkeypatch, filename):
    """Even a semantically harmless fixture rewrite must fail before comparison."""
    tool, report, key = fixture(tmp_path,monkeypatch,np.asarray([1.]))
    with (tmp_path/'reference'/filename).open('ab') as handle:
        handle.write(b' ')
    with pytest.raises(AssertionError,match='frozen fixture changed'):
        tool.compare('O06',report,tolerance='public')


def test_tc_reference_cannot_be_unfrozen_or_overwritten(tmp_path, monkeypatch):
    """Capture refuses either existing file, and comparison requires both hashes."""
    tool, report, key = fixture(tmp_path,monkeypatch,np.asarray([1.]))
    with pytest.raises(AssertionError,match='refusing overwrite'):
        tool.worker('O06','reference')
    manifest = json.loads(tool.MANIFEST.read_text())
    manifest['thermocalc_sampling']['reference_sha256'] = {}
    tool.MANIFEST.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError,match='frozen fixture hashes'):
        tool.compare('O06',report,tolerance='public')


def test_tc_metadata_changes_only_the_cache_field():
    """All text outside the one cache value, including other paths/numbers, is retained."""
    normalize = runner().thermocalc_tools().normalize_metadata
    first = json.dumps(dict(outside='keep/a',thermocalc_config=dict(cache_dir='local/a',flag=1),number=-0.0),sort_keys=True)
    second = first.replace('local/a','different/cache/path')
    expected = first.replace('"local/a"','null')
    assert normalize(np.asarray(first)).item() == expected
    assert normalize(np.asarray(second)).item() == expected
    for changed in [second.replace('keep/a','keep/b'),second.replace('"flag": 1','"flag": 2'),
                    second.replace('-0.0','0.0'),second.replace('"number": ', '"number":  ')]:
        assert normalize(np.asarray(changed)).item() != expected


@pytest.mark.parametrize('change_other_field',[False,True])
def test_tc_metadata_participates_in_compare(tmp_path,monkeypatch,change_other_field):
    """A different cache passes, while any other metadata field still fails the gate."""
    tool,report,key = fixture(tmp_path,monkeypatch,np.asarray([1.]))
    metadata_key = 'provider/metadata_json'
    first = dict(thermocalc_config=dict(cache_dir='first',flag=1),other='retained')
    second = dict(thermocalc_config=dict(cache_dir='second',flag=1),other='changed' if change_other_field else 'retained')
    for owner,metadata in [('reference',first),('package',second)]:
        np.savez(tmp_path/owner/'O06.npz',**{key:np.asarray([1.]),metadata_key:np.asarray(json.dumps(metadata,sort_keys=True))})
    manifest=json.loads(tool.MANIFEST.read_text())
    manifest['thermocalc_sampling']['reference_sha256']['O06.npz']=hashlib.sha256((tmp_path/'reference/O06.npz').read_bytes()).hexdigest()
    tool.MANIFEST.write_text(json.dumps(manifest))
    result=tool.compare('O06',report,tolerance='public')
    assert result.get('failed_keys',[]) == ([metadata_key] if change_other_field else [])
    measured=json.loads((tmp_path/'package/O06.deviations.json').read_text())
    assert measured['bit_identical'] is (not change_other_field)


def test_tc_missing_resources_are_sanitized_and_other_errors_not_hidden():
    """Only known missing resources yield skip reasons; arbitrary failures propagate."""
    classify=runner().thermocalc_tools().unavailable_reason
    assert classify(ModuleNotFoundError('missing',name='tc_python'))=='TC-Python is not installed'
    assert classify(RuntimeError('No valid licence'))=='TC-Python licence unavailable'
    assert classify(ValueError('unrelated setup failure')) is None
