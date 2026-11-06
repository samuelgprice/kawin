"""Numerical edge cases and privacy boundaries for acceptance deviation reports."""
import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest


def metrics():
    """Load reporting functions independently of any provider or solver."""
    spec=importlib.util.spec_from_file_location('metrics_tool',Path(__file__).parents[1]/'tools/acceptance_metrics.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_elementwise_margin_is_not_ratio_of_maxima():
    """The worst relative tolerance utilization can occur away from maximum absolute error."""
    result=metrics().array_metrics(np.array([1.,100.]),np.array([1.01,100.02]),rtol=.001,atol=0.)
    assert result['max_absolute_deviation']==pytest.approx(.02)
    assert result['max_relative_deviation']==pytest.approx(.01)
    assert result['max_tolerance_fraction']==pytest.approx(10.)
    assert result['worst_error']==pytest.approx(.01)
    assert result['worst_allowed_error']==.001


def test_zero_reference_relative_error_is_explicitly_unbounded():
    """A small absolute error can pass atol while having unbounded relative error."""
    result=metrics().array_metrics(np.array([0.]),np.array([1e-13]),rtol=1e-10,atol=1e-12)
    assert result['max_relative_deviation'] is None
    assert result['relative_deviation_unbounded']
    assert result['max_tolerance_fraction']==pytest.approx(.1)
    json.dumps(result,allow_nan=False)


def test_equal_nonfinite_values_are_not_finite_errors():
    """Matching NaN masks and infinities preserve zero error and actual byte identity."""
    array=np.array([np.nan,np.inf,-np.inf,0.])
    result=metrics().array_metrics(array,array.copy(),rtol=0.,atol=0.)
    assert result['bit_identical'] and result['max_absolute_deviation']==0.
    assert result['max_relative_deviation']==0.
    assert not result['within_10x_of_tolerance']


def test_nonfinite_mismatch_is_not_reported_as_zero_error():
    """A changed infinity or mask must remain visibly unbounded in strict JSON."""
    result=metrics().array_metrics(np.array([np.inf]),np.array([-np.inf]),rtol=1e-10,atol=1e-12)
    assert result['max_absolute_deviation'] is None
    assert result['tolerance_fraction_unbounded'] and result['within_10x_of_tolerance']
    json.dumps(result,allow_nan=False)


def test_signed_zero_is_not_bit_identical():
    """Bit identity is stronger than numerical equality even when deviation is zero."""
    result=metrics().array_metrics(np.array([0.]),np.array([-0.]),rtol=1e-10,atol=1e-12)
    assert not result['bit_identical'] and result['max_absolute_deviation']==0.


def test_quantity_retains_different_array_tolerances():
    """Mixed analytic quantities keep their own scales and worst normalized error."""
    tool=metrics()
    arrays={'interface':tool.array_metrics(np.array([1.]),np.array([1.001]),rtol=0.,atol=.01),
            'profile':tool.array_metrics(np.array([1.]),np.array([1.002]),rtol=0.,atol=.1)}
    result=tool.quantity_metrics(arrays)
    assert result['max_absolute_deviation']==pytest.approx(.002)
    assert result['worst_array']=='interface'
    assert result['tolerance']['interface']['atol']==.01
    assert result['tolerance']['profile']['atol']==.1
    compact=tool.public_cases({'C04':{'quantities':{'analytic_error':result}}})
    exported=compact['C04']['quantities']['analytic_error']
    assert 'arrays' not in exported
    assert exported['tolerance']==list(result['tolerance'].values())
    same=tool.quantity_metrics({'first':arrays['interface'],'second':arrays['interface']})
    compact=tool.public_cases({'C01':{'quantities':{'interface':same}}})
    assert compact['C01']['quantities']['interface']['tolerance']['atol']==.01


@pytest.mark.parametrize("case", ["O04", "O06"])
def test_private_metrics_do_not_enter_committed_report(case):
    """Private errors and dimensional scales are redacted even in worst/near summaries."""
    tool=metrics()
    sensitive=8765.4321
    entry=dict(case=case,quantity='interface',error=sensitive,allowance=sensitive)
    summary=dict(bit_identical_runs=0,within_tolerance_not_identical_runs=1,failed_runs=0,
                 worst=entry,within_10x=[entry])
    public=dict(cases=tool.public_cases({case:{'secret':sensitive},'C01':{'public':0.}}),
                summary=tool.public_summary(summary))
    encoded=json.dumps(public,allow_nan=False)
    assert str(sensitive) not in encoded
    assert public['cases'][case]['private_metrics']
    assert public['summary']['worst']==dict(case=case,quantity='interface',private_metrics=True)


def test_overflow_is_explicit_and_serializable():
    """Extreme finite inputs cannot silently create invalid Infinity JSON values."""
    result=metrics().array_metrics(np.array([-1e308]),np.array([1e308]),rtol=0.,atol=1e-12)
    assert result['absolute_deviation_unbounded'] and result['tolerance_fraction_unbounded']
    json.dumps(result,allow_nan=False)



@pytest.mark.parametrize('actual', [0., -0.])
def test_zero_error_has_no_worst_entry(actual, capsys):
    """Identical arrays and signed-zero-only differences must both report worst: none."""
    tool=metrics()
    measured=tool.array_metrics(np.array([0.]),np.array([actual]),rtol=1e-10,atol=1e-12)
    summary=tool.summarize({'C01':dict(passed=True,bit_identical=measured['bit_identical'],
        quantities={'interface':tool.quantity_metrics({'interfaceData/values':measured})})})
    assert summary['worst'] is None
    tool.print_summary('Acceptance',summary)
    assert '(worst: none)' in capsys.readouterr().out
    assert tool.public_summary(summary)['worst'] is None


def test_compact_export_has_one_case_identity_and_no_array_detail():
    """Quantity exports retain the requested aggregate fields and distinct tolerances only."""
    tool=metrics()
    array=tool.array_metrics(np.array([1.]),np.array([1.]),rtol=1e-10,atol=1e-12)
    record=dict(bit_identical=True,passed=True,case='C01',tolerance_mode='public',
        quantities={'interface':tool.quantity_metrics({'private_array_name':array})})
    exported=tool.public_cases({'C01':record})['C01']
    assert set(exported)=={'bit_identical','quantities'}
    assert set(exported['quantities']['interface'])=={
        'max_absolute_deviation','max_relative_deviation','max_tolerance_fraction','tolerance'}
    assert 'private_array_name' not in json.dumps(exported)
