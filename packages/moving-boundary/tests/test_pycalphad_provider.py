"""Installed generation boundaries and the adapter's phase/component contracts."""
import importlib.util
from pathlib import Path
import site
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest


def optional_tool():
    """Load generation guards without loading any research case definition."""
    spec=importlib.util.spec_from_file_location('generation_tool',Path(__file__).parents[1]/'tools/acceptance_optional.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_provider_module_import_is_lazy(tmp_path):
    """Importing helpers must not import kawin or pycalphad, even if both are blocked."""
    import moving_boundary
    code="""
import site, sys
sys.path.insert(0,sys.argv[1])
sys.path.extend(site.getsitepackages())
class NoEngine:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'kawin','pycalphad','examples','debugpy'}:
            raise AssertionError('Engine/research import before generation: '+fullname)
sys.meta_path.insert(0,NoEngine())
from moving_boundary.providers.pycalphad import resolve_pycalphad_equilibrium_phases
assert resolve_pycalphad_equilibrium_phases(None,['FE','CR','NI'],['FCC_A1','BCC_A2'])==('FCC_A1','BCC_A2')
assert 'pycalphad' not in sys.modules and 'kawin' not in sys.modules
"""
    subprocess.run([sys.executable,'-I','-S','-c',code,str(Path(moving_boundary.__file__).parents[1])],
        cwd=tmp_path,stdin=subprocess.DEVNULL,check=True,timeout=30)


def test_phase_universe_preserves_resolved_order():
    """Changing the interface order cannot reorder the full equilibrium phase universe."""
    from moving_boundary.providers.pycalphad import order_pycalphad_phase_universe
    universe=('LIQUID','FCC_A1','BCC_A2','FCC_A1')
    assert order_pycalphad_phase_universe(['BCC_A2','FCC_A1'],universe)==('LIQUID','FCC_A1','BCC_A2')
    assert order_pycalphad_phase_universe(['FCC_A1','BCC_A2'],universe)==('LIQUID','FCC_A1','BCC_A2')
    with pytest.raises(ValueError,match='missing'):
        order_pycalphad_phase_universe(['BCC_A2','HCP_A3'],universe)
    with pytest.raises(ValueError,match='exactly two'):
        order_pycalphad_phase_universe(['BCC_A2'],universe)


@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf])
def test_nonfinite_g_offset_is_rejected(value):
    """Gibbs-energy offsets must remain finite before engine construction is used."""
    from moving_boundary.providers.pycalphad import _validate_g_offset
    with pytest.raises(ValueError,match='finite'):
        _validate_g_offset(value)


def test_endpoint_helpers_preserve_requested_phase_and_input_element_order():
    """Metadata reverses phase records as needed, and excludes the reference/vacancy components."""
    from moving_boundary.providers.pycalphad import PycalphadDefaultPhaseThermodynamics
    facade=PycalphadDefaultPhaseThermodynamics.__new__(PycalphadDefaultPhaseThermodynamics)
    facade.elements=['FE','NI','CR','VA']
    facade.phase_amount_tolerance=1e-12
    # These are composition-set fixtures for ordering helpers, not generation substitutes.
    def composition(phase, values, amount=1.):
        return SimpleNamespace(phase_record=SimpleNamespace(phase_name=phase),X=values,NP=amount)
    records=[composition('FCC_A1',[.2,.5,.3]),composition('BCC_A2',[.1,.7,.2])]
    metadata=facade._metadata_from_composition_sets(records,('BCC_A2','FCC_A1'))
    assert metadata['endpoint_phases']==('BCC_A2','FCC_A1')
    np.testing.assert_array_equal(metadata['endpoints'][0]['composition'],[.2,.1])
    np.testing.assert_array_equal(metadata['endpoints'][1]['composition'],[.3,.2])
    filtered=facade._active_composition_sets([records[0],composition('LIQUID',[.2,.5,.3],1e-13),records[1]])
    assert [item.phase_record.phase_name for item in filtered]==['FCC_A1','BCC_A2']


_GENERATION_GUARD_SCRIPT = r"""
import importlib.machinery, importlib.util, re, site, sys
from pathlib import Path
sys.path.extend(site.getsitepackages())
spec=importlib.util.spec_from_file_location('generation_tool',sys.argv[1])
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
operation,path=sys.argv[2],Path(sys.argv[3])
guard=module.guard_package_generation
try:
    with guard():
        if operation=='import':
            __import__('examples')
        elif operation=='path_read':
            path.read_text()
        else:
            importlib.machinery.SourceFileLoader('forbidden_adapter',str(path)).get_data(str(path))
except AssertionError as exc:
    assert re.search('Research',str(exc)),exc
else:
    raise SystemExit('the generation guard did not raise')
assert path.read_text()=='answer=42'
with guard() as audit:
    pass
assert audit==dict(guarded=True,examples_imported=False,examples_read=False),audit
"""


@pytest.mark.parametrize('operation',['import','path_read','loader_read'])
def test_generation_guard_blocks_research_imports_and_source_reads(tmp_path, operation):
    """The guard catches imports and direct reads, then becomes inert for research assembly.

    Runs in its own process without the M9 independence variables: this negative test imports
    ``examples`` on purpose, which an active ``MB_FORBID_RESEARCH`` guard would refuse first, and
    the generation guard's audit hook cannot be removed from the process afterwards.
    """
    import os
    path=tmp_path/'examples/ternaryExamples/pycalphad_default_phase_adapter.py'
    path.parent.mkdir(parents=True)
    path.write_text('answer=42')
    env={key:value for key,value in os.environ.items() if not key.startswith(('PYTHON','MB_FORBID','MB_GUARD'))}
    result=subprocess.run([sys.executable,'-I','-S','-c',_GENERATION_GUARD_SCRIPT,
                           str(Path(__file__).parents[1]/'tools/acceptance_optional.py'),operation,str(path)],
                          cwd=tmp_path,env=env,capture_output=True,text=True,stdin=subprocess.DEVNULL,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
