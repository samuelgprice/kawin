"""CLI reporting contracts with isolated outputs and no expensive solver jobs."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest


def isolated_runner(tmp_path, monkeypatch, *, failure=False):
    """Exercise real comparison/report orchestration with tiny generated test arrays."""
    project=Path(__file__).parents[1]
    spec=importlib.util.spec_from_file_location('reporting_runner',project/'tools/acceptance.py')
    tool=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    metrics=tool.metric_tools()
    monkeypatch.setattr(tool,'metric_tools',lambda:metrics)
    output=tmp_path/'.verification/acceptance'
    (tmp_path/'docs').mkdir()
    (tmp_path/'.verification').mkdir()
    (tmp_path/'.verification/public-environment.json').write_text(json.dumps({'python':sys.executable}))
    acceptance={'review_followup':{'full':{'deviations':{'worst':dict(fraction=0.,error=0.,unbounded=False)}}}}
    (tmp_path/'docs/acceptance.json').write_text(json.dumps(acceptance))
    (tmp_path/'docs/acceptance-deviations.json').write_text('{"previous": true}')
    manifest=dict(reference_commit='06dc7b4ab56f67226ceabfde41ebf2690c088841',quick=['C01'],
        contracts=[],full_drift=[],thermocalc_sampling=dict(repeat_unseeded=True),
        public_tolerances={'profiles':dict(rtol=1e-10,atol=1e-12)})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(tool,'PROJECT',tmp_path)
    monkeypatch.setattr(tool,'OUTPUT',output)
    monkeypatch.setattr(tool,'MANIFEST',tmp_path/'manifest.json')
    monkeypatch.setattr(tool,'validate_expected',lambda cases:manifest)
    monkeypatch.setattr(tool,'setup_targets',lambda:dict(target='test-target',host='test-host'))
    monkeypatch.setattr(tool,'check_repeatability',lambda *args,**kwargs:None)
    monkeypatch.setattr(tool,'run_pytest',lambda *args,**kwargs:dict(tests=1,failures=0,errors=0,skipped=0))
    def worker(case, role, **kwargs):
        if kwargs.get("probe_thermocalc"):
            return dict(status="PASS",thermocalc_availability=dict(available=False,reason="TC-Python is not installed"))
        if failure and case=='C01':
            raise RuntimeError('deliberate test failure')
        report=dict(status='PASS')
        for owner in ('reference','package'):
            folder=output/owner
            folder.mkdir(exist_ok=True)
            (folder/(case+'.json')).write_text(json.dumps(report))
            np.savez(folder/(case+'.npz'),sample=np.array([0.,1.]))
        return report
    monkeypatch.setattr(tool,'worker',worker)
    return tool


@pytest.mark.parametrize('action', ['quick','full'])
@pytest.mark.parametrize('failure', [False,True])
def test_cli_without_record_leaves_tracked_files_unchanged(tmp_path, monkeypatch, action, failure):
    """Both passing and failing default gates leave every pre-existing project file intact."""
    tool=isolated_runner(tmp_path,monkeypatch,failure=failure)
    tracked={path:path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    assert tool.main([action])==int(failure)
    assert {path:path.read_bytes() for path in tracked}==tracked
    assert (tool.OUTPUT/(action+'-deviations.json')).exists()
    if action == "full":
        summary=json.loads((tool.OUTPUT/"full-summary.json").read_text())
        assert summary["skipped"]["O06"] == "TC-Python is not installed"


@pytest.mark.parametrize('action', ['quick','full'])
def test_record_requires_passing_gate_and_normalizes_zero_worst(tmp_path, monkeypatch, action):
    """Explicit recording uses fresh public metrics, with one record per case and null worst."""
    tool=isolated_runner(tmp_path,monkeypatch)
    assert tool.main([action,'--record'])==0
    document=json.loads((tmp_path/'docs/acceptance-deviations.json').read_text())
    assert document['tier']==action and set(document['cases'])=={'C01','C32','LEGACY'}
    assert document['summary']['worst'] is None
    encoded=json.dumps(document)
    assert str(tmp_path) not in encoded and 'arrays' not in encoded
    evidence=json.loads((tmp_path/'docs/acceptance.json').read_text())
    assert all(entry['deviations']['worst'] is None for entry in evidence['review_followup'].values())


def test_failed_record_run_cannot_replace_committed_report(tmp_path, monkeypatch):
    """A failure before completion must preserve the previous committed evidence bytes."""
    tool=isolated_runner(tmp_path,monkeypatch,failure=True)
    paths=list((tmp_path/'docs').glob('*.json'))
    before={path:path.read_bytes() for path in paths}
    assert tool.main(['quick','--record'])==1
    assert {path:path.read_bytes() for path in paths}==before


@pytest.mark.parametrize('argv', [
    ['historical','capture','--research','x','--record'],['setup','--record'],
    ['quick','--tolerance','exact','--record'],['full','--cases','C01','--record'],
    ['quick','--cases','--record'],
    ['quick','--research','x'],['full','repeat'],['historical','repeat'],['historical','--research','x']])
def test_record_rejects_non_public_or_partial_runs(tmp_path, monkeypatch, argv):
    """Invalid recording requests, and normal/historical argument mixes, fail before setup or any write."""
    tool=isolated_runner(tmp_path,monkeypatch)
    with pytest.raises(SystemExit) as error:
        tool.main(argv)
    assert error.value.code==2
    assert not tool.OUTPUT.exists()


def test_inventory_scale_divides_only_mole_arrays_by_molar_volume():
    """Manifest rule: length scale; divided by minimum explicit Vm for mole inventories only."""
    project=Path(__file__).parents[1]
    spec=importlib.util.spec_from_file_location('inventory_scale_tool',project/'tools/acceptance.py')
    tool=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    explicit=dict(length_scale=2.,minimum_molar_volume=7.2e-6)
    for key in ('molarInventoryData/values','totalMolesData/values','dependentMoleClosureErrorData/values',
                'model3/totalMolesData/values'):
        assert tool.quantity(key)=='inventory'
        assert tool.inventory_scale(key,explicit)==2./7.2e-6
    for key in ('inventoryData/values','state/_initialInventory','concData/values','model0/state/_initialInventory'):
        assert tool.quantity(key)=='inventory'
        assert tool.inventory_scale(key,explicit)==2.
    assert tool.inventory_scale('totalMolesData/values',dict(length_scale=5.))==5.


@pytest.mark.parametrize('record, message', [
    (None, 'O01 input not verified: private-inputs.json missing; cannot verify available input: exb5.json'),
    ({}, 'O01 input not verified: input not hashed in private-inputs.json: exb5.json'),
    ({'exb5.json':{'status':'PRESENT','sha256':{}}}, 'O01 input not verified: input hash record empty or malformed: exb5.json')])
def test_available_private_input_without_valid_record_fails_the_run(tmp_path, monkeypatch, record, message):
    """An available input of an enabled case is never run unverified; absent ones keep their skip."""
    tool=isolated_runner(tmp_path,monkeypatch)
    inputs=tool.OUTPUT/'inputs'
    inputs.mkdir(parents=True)
    (inputs/'exb5.json').write_text('{}')
    if record is not None:
        (inputs/'private-inputs.json').write_text(json.dumps(record))
    with pytest.raises(AssertionError) as error:
        tool.main(['full'])
    assert str(error.value).startswith(message)
    assert not (tool.OUTPUT/'package').exists()


def test_absent_private_inputs_keep_their_documented_skip(tmp_path, monkeypatch):
    tool=isolated_runner(tmp_path,monkeypatch)
    assert tool.main(['full'])==0
    skipped=json.loads((tool.OUTPUT/'full-summary.json').read_text())['skipped']
    assert skipped['O01']=='cached input missing' and skipped['O05']=='database input missing'
