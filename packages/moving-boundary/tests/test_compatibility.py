"""Focused tests for policies deliberately outside public kawin."""
import numpy as np
import pytest


def test_host_loop_is_inherited():
    """Compatibility delegates iteration to the unchanged dependency."""
    from moving_boundary.compatibility import DESolver
    from kawin.solver import DESolver as Host
    assert DESolver.solve.__globals__['_HostSolver'] is Host
    assert '_updateX' not in DESolver.__dict__


@pytest.mark.parametrize('step,remaining,allowed',[(.01,1.,False),(.01,.01,True),(.2,1.,True)])
def test_step_policy(step,remaining,allowed):
    """Reject a genuinely small step; allow an exact final remainder."""
    from moving_boundary.compatibility import DESolver
    solver=DESolver()
    solver._compatFinalTime=remaining
    solver._dtmin,solver._dtmax=.1,1.
    solver._X0=np.array([1.])
    solver._unflattenX=lambda x,template:x
    solver._flattenX=lambda x:x
    solver._f=lambda t,x:np.zeros_like(x)
    solver._getDt=lambda dx:step
    if allowed:
        assert solver._getdXdt(0.,np.array([1.]),True)[1]==step
    else:
        with pytest.raises(ValueError,match='minimum allowed'):
            solver._getdXdt(0.,np.array([1.]),True)


def test_recording_lookup_and_preallocation():
    """Unused preallocated rows cannot change interpolation or exact-time policy."""
    from moving_boundary.compatibility import MeshData
    class Mesh:
        y=np.zeros((2,1))
        def flattenResponse(self,y):
            return y
    data=MeshData(Mesh(),True)
    data.record(0.,np.zeros((2,1)))
    data.record(1.,np.ones((2,1)))
    data.preallocate(2000)
    np.testing.assert_array_equal(data.y(.5),np.full((2,1),.5))
    np.testing.assert_array_equal(data.y(2.),np.ones((2,1)))
    data._timeInterpolationEnabled=False
    with pytest.raises(ValueError,match='exact recorded time'):
        data.y(.5)
    snapshot=data.y(1.)
    snapshot[:]=7.
    np.testing.assert_array_equal(data.y(1.),np.ones((2,1)))


def test_explicit_gas_constant_does_not_patch_host():
    """The local tracer wrapper uses research R and leaves stock constants untouched."""
    import kawin.Constants as constants
    import kawin.thermo.Mobility as mobility
    from moving_boundary.compatibility.thermo import GAS_CONSTANT
    assert GAS_CONSTANT==8.3145
    assert constants.GAS_CONSTANT==8.314
    assert mobility.tracer_diffusivity_from_mobility(1.,1.)==8.314


def test_composition_floor_is_local():
    """Default composition constraints differ only in the package subclass."""
    from moving_boundary.compatibility import DiffusionConstraints
    from kawin.diffusion.DiffusionParameters import DiffusionConstraints as Host
    assert DiffusionConstraints().minComposition==1e-10
    assert Host().minComposition==1e-8
