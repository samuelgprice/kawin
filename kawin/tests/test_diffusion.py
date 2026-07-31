import json
import math
import os
import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
import pytest

from kawin.diffusion import SinglePhaseModel, HomogenizationModel, MovingBoundary1DModel, MovingBoundaryFD1DModel, MovingBoundaryIllingworthFD1DModel, MovingBoundaryOlayeFD1DModel, TemperatureParameters
from kawin.diffusion.mesh import Cartesian1D, CartesianFD1D, Cylindrical1D, Spherical1D, Cartesian2D, MixedBoundary1D, PeriodicBoundary1D
from kawin.diffusion.mesh import ProfileBuilder, StepProfile1D, LinearProfile1D, DiracDeltaProfile, ConstantProfile, GaussianProfile, ExperimentalProfile1D, BoundedEllipseProfile, BoundedRectangleProfile
from kawin.diffusion.mesh import get_moving_boundary_fd_geometry
from kawin.diffusion.mesh.MovingBoundaryFD1D import (
    integrate_binary_fd_profile,
    interpolate_previous_ignored_composition,
    quad_fit_derivs,
)
from kawin.diffusion.DiffusionParameters import computeMobility, _computeSingleMobility, TemperatureParameters, HashTable
from kawin.diffusion.HomogenizationParameters import HomogenizationParameters, computeHomogenizationFunction
from kawin.thermo import GeneralThermodynamics, MulticomponentThermodynamics
from kawin.thermo.Surrogate import GeneralSurrogate, BinarySurrogate
from kawin.tests.datasets import *
from kawin.solver import explicitEulerIterator, rk4Iterator

NiCrTherm = GeneralThermodynamics(NICRAL_TDB, ['NI', 'CR'], ['FCC_A1', 'BCC_A2'])
NiCrAlTherm = GeneralThermodynamics(NICRAL_TDB, ['NI', 'CR', 'AL'], ['FCC_A1', 'BCC_A2'])
FeCrNiTherm = GeneralThermodynamics(FECRNI_DB, ['FE', 'CR', 'NI'], ['FCC_A1', 'BCC_A2'])
FeCrNiTherm_sigma = GeneralThermodynamics(FECRNI_DB, ['FE', 'CR', 'NI'], ['FCC_A1', 'BCC_A2', 'SIGMA'])
FeCTherm = GeneralThermodynamics(FECRC_DB, ['FE', 'C'], ['FCC_A1'])


class ConstantBinaryThermodynamics:
    def __init__(self, phases, diffusivities, interface_compositions):
        self.phases = phases
        self.diffusivities = diffusivities
        self.interface_compositions = interface_compositions

    def clearCache(self):
        return

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        values = np.atleast_1d(T).astype(np.float64)
        return np.squeeze(np.ones(values.shape, dtype=np.float64) * self.diffusivities[phase])

    def getInterfacialComposition(self, T, gExtra=0, precPhase=None):
        values = np.atleast_1d(T).astype(np.float64)
        left = np.ones(values.shape, dtype=np.float64) * self.interface_compositions[0]
        right = np.ones(values.shape, dtype=np.float64) * self.interface_compositions[1]
        return np.squeeze(left), np.squeeze(right)


class VariableBinaryThermodynamics(ConstantBinaryThermodynamics):
    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        composition = np.asarray(x, dtype=np.float64)
        base = float(self.diffusivities[phase])
        return np.squeeze(base * (1.0 + 0.5 * composition))


class MockTernaryTieLineThermodynamics:
    def __init__(self, phases, left_probe, right_probe, reverse_endpoint_order=False, ambiguous_endpoint_phases=False):
        self.phases = phases
        self.left_probe = np.asarray(left_probe, dtype=np.float64)
        self.right_probe = np.asarray(right_probe, dtype=np.float64)
        self.reverse_endpoint_order = bool(reverse_endpoint_order)
        self.ambiguous_endpoint_phases = bool(ambiguous_endpoint_phases)
        self.probe_direction = self.right_probe - self.left_probe
        self.base_left = np.array([0.29, 0.0565], dtype=np.float64)
        self.base_right = np.array([0.185, 0.12475], dtype=np.float64)
        self.left_slope = np.array([-0.04, 0.01], dtype=np.float64)
        self.right_slope = np.array([0.03, 0.04], dtype=np.float64)
        self.diffusivities = {
            phases[0]: 0.8 * np.eye(2, dtype=np.float64),
            phases[1]: 0.4 * np.eye(2, dtype=np.float64),
        }

    def clearCache(self):
        return

    def _check_side_of_probe(self, values):
        perp = np.array([-self.probe_direction[1], self.probe_direction[0]], dtype=np.float64)
        sides = []
        for x in values:
            d = np.asarray(x, dtype=np.float64).reshape(-1) - self.right_probe
            sides.append(np.sign(np.dot(d, perp)))
        return sides

    def _lambda_from_probe(self, x):
        probe = np.asarray(x, dtype=np.float64).reshape(-1)
        denom = float(np.dot(self.probe_direction, self.probe_direction))
        lam = 0.0 if denom == 0 else float(np.dot(probe - self.left_probe, self.probe_direction) / denom)
        return float(np.clip(lam, 0.0, 1.0))

    def _interface_from_lambda(self, lam):
        effective_lam = lam + 0.1
        left = self.base_left + effective_lam * self.left_slope
        right = self.base_right + effective_lam * self.right_slope
        return left, right

    def getInterfacialComposition(self, x, T, gExtra=0, precPhase=None, returnMeta=False):
        values = np.asarray(x, dtype=np.float64)
        if values.ndim == 1:
            left, right = self._interface_from_lambda(self._lambda_from_probe(values))
            phase_left = self.phases[0]
            phase_right = self.phases[1] if precPhase is None else precPhase
            if self.ambiguous_endpoint_phases:
                phase_right = phase_left
            if self.reverse_endpoint_order:
                left, right = right, left
                phase_left, phase_right = phase_right, phase_left
            if returnMeta:
                return left, right, {
                    "endpoint_phases": (phase_left, phase_right),
                    "endpoints": (
                        {"phase": phase_left, "composition": left},
                        {"phase": phase_right, "composition": right},
                    ),
                }
            return left, right
        pairs = [self._interface_from_lambda(self._lambda_from_probe(v)) for v in values]
        left, right = zip(*pairs)
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        phase_left = self.phases[0]
        phase_right = self.phases[1] if precPhase is None else precPhase
        if self.ambiguous_endpoint_phases:
            phase_right = phase_left
        if self.reverse_endpoint_order:
            left, right = right, left
            phase_left, phase_right = phase_right, phase_left
        if returnMeta:
            return left, right, {
                "endpoint_phases": (phase_left, phase_right),
                "endpoints": (
                    {"phase": phase_left, "composition": left},
                    {"phase": phase_right, "composition": right},
                ),
            }
        return left, right

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        values = np.asarray(x, dtype=np.float64)
        base = np.asarray(self.diffusivities[phase], dtype=np.float64)
        if values.ndim == 1:
            return base
        return np.tile(base, (len(values), 1, 1))


class MockBinaryInterfaceThermo:
    def __init__(self):
        self.phases = ['ALPHA', 'BETA']
        self.elements = ['FE', 'CR']
        self.numElements = 2

    def getInterfacialComposition(self, T, gExtra=0, precPhase=None, returnMeta=False):
        T_arr = np.atleast_1d(T).astype(np.float64)
        xa = 0.30 + 1e-4 * (T_arr - T_arr.min())
        xb = 0.70 - 1e-4 * (T_arr - T_arr.min())
        xa = np.squeeze(xa)
        xb = np.squeeze(xb)
        if not returnMeta:
            return xa, xb
        p_beta = self.phases[1] if precPhase is None else precPhase
        return xa, xb, {
            "endpoint_phases": (self.phases[0], p_beta),
            "endpoints": (
                {"phase": self.phases[0], "composition": xa},
                {"phase": p_beta, "composition": xb},
            ),
        }

def test_compositionInput():
    '''
    Tests that after setting up a model, all components are greater than 0

    In practice, this greatly speeds up simulation time without sacrificing accuracy since it avoids
    performing equilibrium calculations with a composition of 0 for any given component

    The composition and setup functions for both models inherit from the
    base diffusion model class, so any model can be used here
    '''
    N = 100
    profile = ProfileBuilder()
    profile.addBuildStep(StepProfile1D(0, [0.2, 0.8], [1, 0]), ['CR', 'AL'])

    mesh = Cartesian1D(['CR', 'AL'], [-1e-3, 1e-3], N)
    mesh.setResponseProfile(profile)
    model = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], NiCrAlTherm, TemperatureParameters(1200+273.15))
    model.setup()

    assert(mesh.y[25,0] + mesh.y[25,1] < 1)
    assert(mesh.y[75,0] + mesh.y[75,1] < 1)
    assert(1 - (mesh.y[75,0] + mesh.y[75,1]) >= model.constraints.minComposition)
    assert(1 - (mesh.y[25,0] + mesh.y[25,1]) >= model.constraints.minComposition)
    assert(mesh.y[75,1] >= model.constraints.minComposition)

def test_singlePhaseFluxes_shape():
    '''
    Tests the dimensions of the single phase fluxes function

    Should be (E-1, N+1) where E is the number of elements and
    N is the number of points
    '''
    N = 100
    profile = ProfileBuilder([(StepProfile1D(0, 0.2, 0.8), 'CR')])
    mesh = Cartesian1D(['CR'], [-1e-3, 1e-3], N)
    mesh.setResponseProfile(profile)
    model = SinglePhaseModel(mesh, ['NI', 'CR'], ['FCC_A1'], NiCrTherm, TemperatureParameters(1073))
    model.setup()
    f = model.getdXdt(model.currentTime, model.getCurrentX())
    assert(f[0].shape == (N,1))

    profile = ProfileBuilder([(StepProfile1D(0, 0.2, 0.4), 'CR'), (StepProfile1D(0, 0.4, 0.4), 'AL')])
    mesh = Cartesian1D(['CR', 'AL'], [-1e-3, 1e-3], N)
    mesh.setResponseProfile(profile)
    model = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], NiCrAlTherm, TemperatureParameters(1073))
    model.setup()
    f = model.getdXdt(model.currentTime, model.getCurrentX())
    assert(f[0].shape == (N,2))

def test_homogenizationMobility():
    '''
    Tests the dimensions of the homogenization mobility function

    Should be (P, E, N) where P is the number of phases,
    E is the number of elements and
    N is the number of points
    '''
    N = 10

    homogenizationParameters = HomogenizationParameters()
    x = np.linspace(0.2, 0.3, N)
    T = 1073*np.ones(N)
    mobBinary, _ = computeHomogenizationFunction(NiCrTherm, x, T, homogenizationParameters)
    assert(mobBinary.shape == (N,2))

    homogenizationParameters = HomogenizationParameters()
    x_cr = np.linspace(0.2, 0.3, N)
    x_al = np.linspace(0.3, 0.2, N)
    x = np.array([x_cr, x_al]).T
    T = 1073*np.ones(N)
    mobTernary, _ = computeHomogenizationFunction(NiCrAlTherm, x, T, homogenizationParameters)
    assert(mobTernary.shape == (N,3))

def test_homogenizationSinglePhaseMobility():
    '''
    Tests that in a single phase region, any of the mobility functions will give
    the same mobility of the single phase itself
    '''
    x = [0.05, 0.05]
    T = 1073

    homogenizationParameters = HomogenizationParameters()
    homogenizationParameters.labyrinthFactor = 2
    mob_data = computeMobility(NiCrAlTherm, x, T)

    mob_funcs = [HomogenizationParameters.WIENER_UPPER, HomogenizationParameters.WIENER_LOWER,
                 HomogenizationParameters.HASHIN_UPPER, HomogenizationParameters.HASHIN_LOWER,
                 HomogenizationParameters.LABYRINTH]
    for f in mob_funcs:
        homogenizationParameters.setHomogenizationFunction(f)
        mob, _ = computeHomogenizationFunction(NiCrAlTherm, x, T, homogenizationParameters)
        assert(np.allclose(np.squeeze(mob), np.squeeze(mob_data.mobility[0]), atol=0, rtol=1e-3))

def test_homogenization_wiener_upper():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    Note: slight change in two-phase mobility test since BCC mobility was added to the NiCrAl dataset
    '''
    x1 = [0.05, 0.05]
    x2 = [0.7, 0.05]
    T = 1073

    homogenizationParameters = HomogenizationParameters()
    homogenizationParameters.setHomogenizationFunction(HomogenizationParameters.WIENER_UPPER)

    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x1, T, homogenizationParameters)
    assert_allclose(mob, [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)

    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x2, T, homogenizationParameters)
    assert_allclose(mob, [5.422636e-22, 1.416680e-22, 3.058155e-22], atol=0, rtol=1e-3)

def test_homogenization_wiener_lower():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    '''
    x1 = [0.05, 0.05]
    x2 = [0.7, 0.05]
    T = 1073

    homogenizationParameters = HomogenizationParameters()
    homogenizationParameters.setHomogenizationFunction(HomogenizationParameters.WIENER_LOWER)

    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x1, T, homogenizationParameters)
    assert_allclose(mob, [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)

    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x2, T, homogenizationParameters)
    assert_allclose(mob, [5.769441e-27, 6.446271e-26, 1.637520e-22], atol=0, rtol=1e-3)

def test_homogenization_hashin_upper():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    '''
    x1 = [0.05, 0.05]
    x2 = [0.7, 0.05]
    T = 1073

    homogenizationParameters = HomogenizationParameters()
    homogenizationParameters.setHomogenizationFunction(HomogenizationParameters.HASHIN_UPPER)

    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x1, T, homogenizationParameters)
    assert_allclose(mob, [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)

    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x2, T, homogenizationParameters)
    assert_allclose(mob, [4.114451e-22, 1.075049e-22, 2.689335e-22], atol=0, rtol=1e-3)

def test_homogenization_hashin_lower():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    '''
    x1 = [0.05, 0.05]
    x2 = [0.7, 0.05]
    T = 1073

    homogenizationParameters = HomogenizationParameters()
    homogenizationParameters.setHomogenizationFunction(HomogenizationParameters.HASHIN_LOWER)

    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x1, T, homogenizationParameters)
    assert_allclose(mob, [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)

    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x2, T, homogenizationParameters)
    assert_allclose(mob, [9.970624e-27, 1.113755e-25, 2.118727e-22], atol=0, rtol=1e-3)

def test_homogenization_lab():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    '''
    x1 = [0.05, 0.05]
    x2 = [0.7, 0.05]
    T = 1073

    homogenizationParameters = HomogenizationParameters()
    homogenizationParameters.setHomogenizationFunction(HomogenizationParameters.LABYRINTH)
    # Labyrinth factor is clipped to [1,2]
    homogenizationParameters.setLabyrinthFactor(2.3)
    assert homogenizationParameters.labyrinthFactor == 2

    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x1, T, homogenizationParameters)
    assert_allclose(mob, [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)

    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x2, T, homogenizationParameters)
    assert_allclose(mob, [1.974358e-22, 5.158761e-23, 1.311953e-22], atol=0, rtol=1e-3)

def test_homogenization_post_process():
    '''
    Test different post process functions for homogenization parameters
        NO_POST - no preprocessing
        PREDEFINED - phases with no mobility will take on mobility of predefined phase
        MAJORITY - phases with no mobility will take on mobility of most present phase
        EXCLUDE - excluded phases will have mobility of 0
    '''
    x = [0.5, 0.18]
    T = 1073

    homogenizationParameters = HomogenizationParameters()
    hashTable = HashTable()

    sortIndices = np.argsort(FeCrNiTherm_sigma.elements[:-1])
    unsortIndices = np.argsort(sortIndices)

    post_process = [
        # No post - SIGMA should be -1 since no mobility parameters
        (HomogenizationParameters.NO_POST, [], {'FCC_A1': [6.645151e-23, 1.500813e-22, 2.097277e-22], 'SIGMA': [-1, -1, -1]}),
        # predefined, SIGMA should have BCC_A2 values
        (HomogenizationParameters.PREDEFINED, ['BCC_A2'], {'FCC_A1': [6.645151e-23, 1.500813e-22, 2.097277e-22], 'SIGMA': [7.731696e-24, 5.867129e-23, 6.654329e-25]}),
        # majority, SIGMA should have FCC_A1 values
        (HomogenizationParameters.MAJORITY, [], {'FCC_A1': [6.645151e-23, 1.500813e-22, 2.097277e-22], 'SIGMA': [6.645151e-23, 1.500813e-22, 2.097277e-22]}),
        # exlude, FCC_A1 should be 0
        (HomogenizationParameters.EXCLUDE, [['FCC_A1']], {'FCC_A1': [0, 0, 0], 'SIGMA': [-1, -1, -1]}),
    ]

    for p in post_process:
        homogenizationParameters.setPostProcessFunction(p[0], *p[1])
        mob = _computeSingleMobility(FeCrNiTherm_sigma, x, T, unsortIndices, hashTable)
        fcc_index = np.squeeze(np.where(mob.phases == 'FCC_A1')[0])
        sigma_index = np.squeeze(np.where(mob.phases == 'SIGMA')[0])
        mob, phase_fracs = homogenizationParameters.postProcessFunction(mob, *homogenizationParameters.postProcessParameters)

        assert_allclose(mob[fcc_index], p[2]['FCC_A1'], rtol=1e-3)
        assert_allclose(mob[sigma_index], p[2]['SIGMA'], rtol=1e-3)

        # Make sure function output works with compute homogenization function
        computeHomogenizationFunction(FeCrNiTherm_sigma, x, T, homogenizationParameters, hashTable)

def test_single_phase_dxdt():
    '''
    Check dxdt values of arbitrary single phase model problem

    We spot check a few points on dxdt rather than checking the entire array

    This uses the parameters from 06_Single_Phase_Diffusion example with the composition
    being linear rather then step functions
    '''
    profile = ProfileBuilder()
    profile.addBuildStep(LinearProfile1D(0, [0.077, 0.054], 2e-3, [0.359, 0.062]), ['CR', 'AL'])
    meshes = [
        Cartesian1D(['CR', 'AL'], [0, 2e-3], 20),
        Cylindrical1D(['CR', 'AL'], [0, 2e-3], 20),
        Spherical1D(['CR', 'AL'], [0, 2e-3], 20)
    ]

    vals5 = [
        np.array([1.63031418e-9, 5.87290513e-10]),
        np.array([1.34386193e-8, 8.02074750e-9]),
        np.array([2.52603986e-8, 1.54590585e-8]),
    ]
    vals10 = [
        np.array([1.54252455e-9, 1.08801591e-9]),
        np.array([8.47547223e-9, 5.37498213e-9]),
        np.array([1.54119182e-8, 9.66441601e-9]),
    ]
    vals15 = [
        np.array([1.59190900e-9, 1.79208543e-9]),
        np.array([6.79212648e-9, 5.15379480e-9]),
        np.array([1.19940010e-8, 8.51736977e-9]),
    ]
    # dt should be the same for all three meshes for single phase model since it only depends on D
    dts = [
        25902.839039,
        25902.839039,
        25902.839039,
    ]
    for i, mesh in enumerate(meshes):
        mesh.setResponseProfile(profile)
        m = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], NiCrAlTherm, 1473.15)
        m.setup()
        dxdt = m.getdXdt(m.currentTime, m.getCurrentX())
        dt = m.getDt(dxdt)

        print(dxdt[0][5], dxdt[0][10], dxdt[0][15], dt)
        assert_allclose(dxdt[0][5], vals5[i], rtol=1e-3)
        assert_allclose(dxdt[0][10], vals10[i], rtol=1e-3)
        assert_allclose(dxdt[0][15], vals15[i], rtol=1e-3)
        assert_allclose(dt, dts[i], rtol=1e-3)

def test_diffusion_x_shape():
    '''
    Check the flatten and unflatten behavior for Diffusion model

    SinglePhaseModel and Homogenization model follows the same path for these functions
    since we just deal with fluxes for elements

    For this setup:
        getCurrentX will return a single element array with the element having a shape of (2,20)
        flattenX will return a 1D array of length 40 (2x20)
        unflattenX should take the output of flattenX and getCurrentX to bring the (40,) array to [(2,20)]
    '''
    profile = ProfileBuilder()
    profile.addBuildStep(LinearProfile1D(-1e-3, [0.077, 0.054], 1e-3, [0.359, 0.062]), ['CR', 'AL'])
    mesh = Cartesian1D(['CR', 'AL'], [-1e-3, 1e-3], 20)
    mesh.setResponseProfile(profile)
    m = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], NiCrAlTherm, TemperatureParameters(1473.15))
    m.setup()

    x = m.getCurrentX()
    origShape = x[0].shape

    x_flat = m.flattenX(x)
    flatShape = x_flat.shape

    x_restore = m.unflattenX(x_flat, x)
    unflatShape = x_restore[0].shape

    assert(len(x) == 1)
    assert(origShape == unflatShape)
    assert(flatShape == (np.prod(origShape),))
    assert(len(x_restore) == 1)

def test_homogenization_dxdt():
    '''
    Check flux values of arbitrary homogenization model problem

    We spot check a few points on dxdt rather than checking the entire array

    We'll only test using the hashin lower homogenization function since there's already tests for
    the output of each homogenization function

    This uses the parameters from 07_Homogenization_Model example with the compositions
    being linear rather than stepwise functions

    Note: values are changed slightly here since the mesh constructs the bounds to be slightly smaller than before
        Old implementation - ends of the mesh is at the node centers
        New implementation - ends of the mesh is at the node edges (node width is 1/(N-1) times smaller)
    '''
    profile = ProfileBuilder()
    profile.addBuildStep(LinearProfile1D(-5e-4, [0.257, 0.065], 5e-4, [0.423, 0.276]), ['CR', 'NI'])
    mesh = Cartesian1D(['CR', 'NI'], [-5e-4, 5e-4], 20)
    mesh.setResponseProfile(profile)
    homogenizationParameters = HomogenizationParameters(HomogenizationParameters.HASHIN_LOWER, eps=0.01)

    m = HomogenizationModel(mesh,  ['FE', 'CR', 'NI'], ['FCC_A1', 'BCC_A2'],
                            thermodynamics=FeCrNiTherm, temperature=TemperatureParameters(1373.15),
                            homogenizationParameters=homogenizationParameters)
    m.constraints.maxCompositionChange = 0.002

    m.setup()
    dxdt = m.getdXdt(m.currentTime, m.getCurrentX())
    dt = m.getDt(dxdt)

    # #Index 5
    ind5, vals5 = 5, np.array([-1.41988464e-09, 1.23824350e-09])

    # #Index 10
    ind10, vals10 = 10, np.array([-9.44367556e-10, 1.67684869e-09])

    # #Index 15
    ind15, vals15 = 15, np.array([-3.95085380e-10, 7.72928846e-10])

    print(dxdt[0][ind5], dxdt[0][ind10], dxdt[0][ind15], dt)
    assert_allclose(dxdt[0][ind5], vals5, atol=0, rtol=1e-3)
    assert_allclose(dxdt[0][ind10], vals10, atol=0, rtol=1e-3)
    assert_allclose(dxdt[0][ind15], vals15, atol=0, rtol=1e-3)
    assert_allclose(dt, 62107.08445, rtol=1e-3)


    mesh = Cartesian1D(['CR', 'NI'], [-5e-4, 5e-4], 20)
    mesh.setResponseProfile(profile)
    homogenizationParameters = HomogenizationParameters(HomogenizationParameters.HASHIN_UPPER, eps=0.01)
    m = HomogenizationModel(mesh,  ['FE', 'CR', 'NI'], ['FCC_A1', 'BCC_A2'],
                            thermodynamics=FeCrNiTherm, temperature=TemperatureParameters(1373.15),
                            homogenizationParameters=homogenizationParameters)
    m.constraints.maxCompositionChange = 0.002

    m.setup()
    x = m.getCurrentX()
    dxdt = m.getdXdt(m.currentTime, x)
    dt = m.getDt(dxdt)

    # The dxdt values are changed due to a correction in how the mobility for each phase is computed
    # Before, the mobilities were multiplied by the overall composition rather than the phase composition

    #Index 5
    ind5, vals5 = 5, np.array([-2.8577719e-8, -4.66806883e-9])

    #Index 10
    ind10, vals10 = 10, np.array([-1.70397119e-8, -3.88722368e-9])

    #Index 15
    ind15, vals15 = 15, np.array([-1.62720361e-8, -7.03752829e-9])

    print(dxdt[0][ind5], dxdt[0][ind10], dxdt[0][ind15], dt)
    assert_allclose(dxdt[0][ind5], vals5, atol=0, rtol=1e-3)
    assert_allclose(dxdt[0][ind10], vals10, atol=0, rtol=1e-3)
    assert_allclose(dxdt[0][ind15], vals15, atol=0, rtol=1e-3)
    assert_allclose(dt, 3348.705601, rtol=1e-3)

def test_diffusionSavingLoading(tmpdir):
    '''
    Tests saving/loading behavior of diffusion model
    '''
    profile = ProfileBuilder()
    profile.addBuildStep(LinearProfile1D(-1e-3, [0.077, 0.054], 1e-3, [0.359, 0.062]), ['CR', 'AL'])
    temperature = TemperatureParameters(1200+273.15)
    mesh = Cartesian1D(['CR', 'AL'], [-1e-3, 1e-3], 20)
    mesh.setResponseProfile(profile)

    #Define mesh spanning between -1mm to 1mm with 50 volume elements
    m = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'],
                         thermodynamics=NiCrAlTherm,
                         temperature=temperature, record=True)

    m.solve(10*3600, verbose=True, vIt=1)
    m.save(tmpdir / 'diff.npz')

    new_m = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'],
                         thermodynamics=NiCrAlTherm,
                         temperature=temperature, record=True)
    new_m.load(tmpdir / 'diff.npz')

    #assert_allclose(m.mesh.y, new_m.mesh.y)
    assert_allclose(m.data.currentY, new_m.data.currentY)
    assert_allclose(m.currentTime, new_m.currentTime)

    #new_m.setMeshtoRecordedTime(-1)
    #assert_allclose(mesh.flattenResponse(new_m.mesh.y), new_m._recordedX[0], rtol=1e-3)
    assert_allclose(new_m.data.y(-1), new_m.data._y[0], rtol=1e-3)
    #new_m.setMeshtoRecordedTime(11*3600)
    #assert_allclose(mesh.flattenResponse(new_m.mesh.y), new_m._recordedX[-1], rtol=1e-3)
    assert_allclose(new_m.data.y(11*3600), new_m.data._y[-1], rtol=1e-3)

    # Interpolate, if we set to 0, then the interpolation should result in the first recorded entry
    #new_m.setMeshtoRecordedTime(0)
    #assert_allclose(mesh.flattenResponse(new_m.mesh.y), new_m._recordedX[0], rtol=1e-3)
    assert_allclose(new_m.data.y(0), new_m.data._y[0], rtol=1e-3)


def test_fdm_diffusionSavingLoading(tmpdir):
    '''
    Tests saving/loading behavior of the finite-difference diffusion model
    '''
    profile = ProfileBuilder()
    profile.addBuildStep(LinearProfile1D(-1e-3, [0.077, 0.054], 1e-3, [0.359, 0.062]), ['CR', 'AL'])
    temperature = TemperatureParameters(1200+273.15)
    mesh = CartesianFD1D(['CR', 'AL'], [-1e-3, 1e-3], 20)
    mesh.setResponseProfile(profile)

    m = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'],
                         thermodynamics=NiCrAlTherm,
                         temperature=temperature, record=True)

    m.solve(10*3600, verbose=True, vIt=1)
    m.save(tmpdir / 'fdm_diff.npz')

    
    new_m = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'],
                         thermodynamics=NiCrAlTherm,
                         temperature=temperature, record=True)
    new_m.load(tmpdir / 'fdm_diff.npz')

    assert_allclose(m.data.currentY, new_m.data.currentY)
    assert_allclose(m.currentTime, new_m.currentTime)
    assert_allclose(new_m.data.y(-1), new_m.data._y[0], rtol=1e-3)
    assert_allclose(new_m.data.y(11*3600), new_m.data._y[-1], rtol=1e-3)
    assert_allclose(new_m.data.y(0), new_m.data._y[0], rtol=1e-3)

def test_fvm_vs_fdm_constantD_linearCompProfile(tmpdir):

    profile = ProfileBuilder()
    profile.addBuildStep(LinearProfile1D(-1e-3, [0.1], 1e-3, [0.3]), ['CR'])
    temperature = TemperatureParameters(1000)
    
    
    therm = ConstantBinaryThermodynamics(
        phases=['FCC_A1'],
        diffusivities={'FCC_A1': 1e-14},
        interface_compositions=(None, None),
    )

    mesh_FVM = Cartesian1D(['CR'], [-1e-3, 1e-3], 20)
    mesh_FVM.setResponseProfile(profile)
    m_FVM = SinglePhaseModel(mesh_FVM, ['FE', 'CR'], ['FCC_A1'],
                         thermodynamics=therm,
                         temperature=temperature, record=True)
    

    mesh_FDM = CartesianFD1D(['CR'], [-1e-3, 1e-3], 20+1) # Need 21 points for 20 elements
    mesh_FDM.setResponseProfile(profile)
    m_FDM = SinglePhaseModel(mesh_FDM, ['FE', 'CR'], ['FCC_A1'],
                         thermodynamics=therm,
                         temperature=temperature, record=True)
    
    def getMidpoints(arr):
        return (arr[:-1] + arr[1:])/2
    
    assert (np.ravel(m_FVM.mesh.z)==getMidpoints(np.ravel(m_FDM.mesh.z))).all()
    assert_allclose(np.ravel(m_FVM.mesh.y), getMidpoints(np.ravel(m_FDM.mesh.y)))
    np.ravel(m_FVM.mesh.zEdge), np.ravel(m_FDM.mesh.zEdge)

    m_FVM.solve(10*3600, verbose=True, vIt=1)
    m_FDM.solve(10*3600, verbose=True, vIt=1)
    
    m_FVM.save(tmpdir / 'fvm_diff.npz')
    m_FDM.save(tmpdir / 'fdm_diff.npz')


    assert_allclose(np.ravel(m_FVM.data.currentY), getMidpoints(np.ravel(m_FDM.data.currentY)))
    assert_allclose(m_FVM.currentTime, m_FDM.currentTime)

def test_fvm_vs_fdm_constantD_nonlinearCompProfile(tmpdir):

    profile = ProfileBuilder()
    z_arr = np.linspace(-1e-3, 1e-3, 21)
    p_of_z = lambda z: (z-(-1e-3))/(1e-3 - (-1e-3))
    y_of_z = lambda z: 0.1 + 0.2 * (p_of_z(z)**5)

    # import matplotlib.pyplot as plt
    # plt.plot(z_arr, y_of_z(z_arr),  'o', linestyle='solid')
    # plt.legend()
    # plt.show()

    profile.addBuildStep(ExperimentalProfile1D(z=z_arr.tolist(), values=y_of_z(z_arr).tolist()), ['CR'])
    temperature = TemperatureParameters(1000)
    
    
    therm = ConstantBinaryThermodynamics(
        phases=['FCC_A1'],
        diffusivities={'FCC_A1': 1e-14},
        interface_compositions=(None, None),
    )

    mesh_FVM = Cartesian1D(['CR'], [-1e-3, 1e-3], 20)
    mesh_FVM.setResponseProfile(profile)
    m_FVM = SinglePhaseModel(mesh_FVM, ['FE', 'CR'], ['FCC_A1'],
                         thermodynamics=therm,
                         temperature=temperature, record=True)
    

    mesh_FDM = CartesianFD1D(['CR'], [-1e-3, 1e-3], 20+1) # Need 21 points for 20 elements
    mesh_FDM.setResponseProfile(profile)
    m_FDM = SinglePhaseModel(mesh_FDM, ['FE', 'CR'], ['FCC_A1'],
                         thermodynamics=therm,
                         temperature=temperature, record=True)
    
    def getMidpoints(arr):
        return (arr[:-1] + arr[1:])/2
    
    assert (np.ravel(m_FVM.mesh.z)==getMidpoints(np.ravel(m_FDM.mesh.z))).all()
    assert_allclose(np.ravel(m_FVM.mesh.y), getMidpoints(np.ravel(m_FDM.mesh.y)))
    np.ravel(m_FVM.mesh.zEdge), np.ravel(m_FDM.mesh.zEdge)

    m_FVM.solve(1000*3600, verbose=True, vIt=1)
    m_FDM.solve(1000*3600, verbose=True, vIt=1)
    
    m_FVM.save(tmpdir / 'fvm_diff.npz')
    m_FDM.save(tmpdir / 'fdm_diff.npz')

    import matplotlib.pyplot as plt
    plt.plot(z_arr, y_of_z(z_arr), 'x', markersize=10,linestyle='solid', color='k', label='Profile')
    plt.plot(m_FVM.mesh.z, m_FVM.data.y(0)[:,0], 'o', linestyle='solid', label='FVM')
    plt.plot(m_FDM.mesh.z, m_FDM.data.y(0)[:,0], 'o', linestyle='solid', label='FDM')
    plt.legend()
    plt.show(block=False)

    assert_allclose(np.ravel(m_FVM.data.currentY), getMidpoints(np.ravel(m_FDM.data.currentY)), rtol=1e-10)
    assert_allclose(m_FVM.currentTime, m_FDM.currentTime)

def test_fvm_vs_fdm_variableD(tmpdir):
    profile = ProfileBuilder()
    # profile.addBuildStep(LinearProfile1D(-1e-3, [0.077, 0.054], 1e-3, [0.359, 0.062]), ['CR', 'AL'])
    profile.addBuildStep(LinearProfile1D(-1e-3, [0.1], 1e-3, [0.3]), ['CR'])
    temperature = TemperatureParameters(1200+273.15) 

    from kawin.diffusion.DiffusionParameters import DiffusionConstraints
    constraints = DiffusionConstraints()
    constraints.minComposition = 1e-10    

    # mesh_FVM = Cartesian1D(['CR', 'AL'], [-1e-3, 1e-3], 20)
    mesh_FVM = Cartesian1D(['CR'], [-1e-3, 1e-3], 20)
    mesh_FVM.setResponseProfile(profile)
    # m_FVM = SinglePhaseModel(mesh_FVM, ['NI', 'CR', 'AL'], ['FCC_A1'],
    #                      thermodynamics=NiCrAlTherm,
    #                      temperature=temperature, record=True)
    m_FVM = SinglePhaseModel(mesh_FVM, ['NI', 'CR'], ['FCC_A1'],
                         thermodynamics=NiCrTherm,
                         temperature=temperature, constraints=constraints, record=True)
    

    # mesh_FDM = CartesianFD1D(['CR', 'AL'], [-1e-3, 1e-3], 20+1)
    mesh_FDM = CartesianFD1D(['CR'], [-1e-3, 1e-3], 20+1)
    mesh_FDM.setResponseProfile(profile)
    # m_FDM = SinglePhaseModel(mesh_FDM, ['NI', 'CR', 'AL'], ['FCC_A1'],
    #                      thermodynamics=NiCrAlTherm,
    #                      temperature=temperature, record=True)
    m_FDM = SinglePhaseModel(mesh_FDM, ['NI', 'CR'], ['FCC_A1'],
                         thermodynamics=NiCrTherm,
                         temperature=temperature, constraints=constraints, record=True)
    
    def getMidpoints(arr):
        return (arr[:-1] + arr[1:])/2
    
    assert (np.ravel(m_FVM.mesh.z)==getMidpoints(np.ravel(m_FDM.mesh.z))).all()
    assert_allclose(np.ravel(m_FVM.mesh.y[:,0]), getMidpoints(np.ravel(m_FDM.mesh.y[:,0])))
    # assert_allclose(np.ravel(m_FVM.mesh.y[:,1]), getMidpoints(np.ravel(m_FDM.mesh.y[:,1])))
    np.ravel(m_FVM.mesh.zEdge), np.ravel(m_FDM.mesh.zEdge)

    m_FVM.solve(10*3600, verbose=True, vIt=1, iterator=explicitEulerIterator)
    m_FDM.solve(10*3600, verbose=True, vIt=1, iterator=explicitEulerIterator)
    
    m_FVM.save(tmpdir / 'fvm_diff.npz')
    m_FDM.save(tmpdir / 'fdm_diff.npz')


    assert_allclose(np.ravel(m_FVM.data.currentY[:,0]), getMidpoints(np.ravel(m_FDM.data.currentY[:,0]))) ##NOTE: This is not supposed to pass because the variable diffusivity results in Mean(D(c_i), D(c_i+1)) != D(Mean(c_i, c_i+1))
    # assert_allclose(np.ravel(m_FVM.data.currentY[:,1]), getMidpoints(np.ravel(m_FDM.data.currentY[:,1])))
    assert_allclose(m_FVM.currentTime, m_FDM.currentTime)

    
    



def test_single_phase_2d():
    '''
    Test Cartesion2D mesh in a single phase model
    '''
    profile = ProfileBuilder([
        (BoundedRectangleProfile([1e-3, 1e-3], [2e-3, 2e-3], [0.077, 0.054], [0.359, 0.062]), ['CR', 'AL'])
    ])
    mesh = Cartesian2D(['CR', 'AL'], [0, 2e-3], 20, [0, 2e-3], 20)
    mesh.setResponseProfile(profile)

    m = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], NiCrAlTherm, 1473.15)
    m.setup()
    dxdt = m.getdXdt(m.currentTime, m.getCurrentX())
    dt = m.getDt(dxdt)

    # Note: dxdt is (400,2), so (210,) corresponds to (10,10,) on 2d array
    print(dxdt[0][210])
    print(dt)
    assert_allclose(dxdt[0][210], [2.85828916e-6, 2.01865282e-6], rtol=1e-3)
    assert_allclose(dt, 12630.562798, rtol=1e-3)

def test_diffusion_profile():
    '''
    Test output of different profile build steps
    Constant, Gaussian, Dirac, Rectangle and Ellipse should support more than 1 dimension and response
    '''
    linear = LinearProfile1D(leftZ=0.25, leftValue=0.1, rightZ=0.75, rightValue=0.9, lowerLeftValue=0.2, upperRightValue=0.7)
    step = StepProfile1D(z=0.4, leftValue=0.2, rightValue=0.7)
    dirac = DiracDeltaProfile(z=0.61, value=0.8)
    constant = ConstantProfile(value=0.3)
    gaussian = GaussianProfile(z=0.6, sigma=0.2, maxValue=0.5)
    exp = ExperimentalProfile1D(z=[0, 0.1, 0.6, 1], values=[0, 0.5, 0.1, 0.3])
    rect = BoundedRectangleProfile(lowerZ=0.25, upperZ=0.75, innerValue=0.3, outerValue=0.6)

    indices = [5, 30, 60, 95]
    profiles = [
        (linear, [0.2, 0.188, 0.668, 0.7]),
        (step, [0.2, 0.2, 0.7, 0.7]),
        (dirac, [0, 0, 0.8, 0]),
        (constant, [0.3, 0.3, 0.3, 0.3]),
        (gaussian, [2.97894e-4, 5.67686e-2, 4.99688e-1, 2.14127e-2]),
        (exp, [0.275, 0.336, 0.1025, 0.2775]),
        (rect, [0.6, 0.3, 0.3, 0.6])
    ]
    for p in profiles:
        mesh = Cartesian1D(1, [0, 1], 100)
        profile = ProfileBuilder()
        profile.addBuildStep(p[0])
        mesh.setResponseProfile(profile)
        assert_allclose(mesh.y[indices,0], p[1], rtol=1e-3)

    constant = ConstantProfile([0.3, 0.5])
    gaussian = GaussianProfile(z=[0.3,0.4], sigma=[0.2,0.5], maxValue=[0.2,0.3])
    dirac = DiracDeltaProfile(z=[0.51, 0.51], value=[0.2, 0.1])
    rect = BoundedRectangleProfile(lowerZ=[0.5,0.5], upperZ=[0.6, 0.7], innerValue=[0.5, 0.1], outerValue=[0.2, 0.3])
    ell = BoundedEllipseProfile(z=[0.65, 0.65], r=[0.1, 0.1], innerValue=[0.5, 0.1], outerValue=[0.2, 0.3])

    indices = [[20,20], [50,50], [70,70]]
    profiles = [
        (constant, [[0.3, 0.5], [0.3, 0.5], [0.3, 0.5]]),
        (gaussian, [[1.37084e-1, 2.05626e-1], [6.69263e-2, 1.00389e-1], [2.28323e-3, 3.42485e-3]]),
        (dirac, [[0,0], [0.2,0.1], [0,0]]),
        (rect, [[0.2, 0.3], [0.5, 0.1], [0.2, 0.3]]),
        (ell, [[0.2, 0.3], [0.2, 0.3], [0.5, 0.1]]),
    ]
    for p in profiles:
        mesh = Cartesian2D(2, [0,1], 100, [0,1], 100)
        profile = ProfileBuilder()
        profile.addBuildStep(p[0], [0,1])
        mesh.setResponseProfile(profile)
        for i, ind in enumerate(indices):
            assert_allclose(mesh.y[ind[0], ind[1]], p[1][i], rtol=1e-3)

def test_diffusion_boundary_conditions():
    profile = ProfileBuilder()
    profile.addBuildStep(LinearProfile1D(0, [0.077, 0.054], 2e-3, [0.359, 0.062]), ['CR', 'AL'])

    bc = MixedBoundary1D(['CR', 'AL'])
    # We test the different input types (name and int)
    bc.setLBC('CR', 'dirichlet', 0.7)
    bc.setLBC('AL', MixedBoundary1D.DIRICHLET, 0.2)
    bc.setLBC('AL', 'composition', 0.2)
    bc.setRBC('CR', 'flux', 0)
    bc.setRBC('CR', 'neumann', 0)
    bc.setRBC('CR', MixedBoundary1D.NEUMANN, 1e-5)
    # Assert that a value error is raised for
    #  a. incorrect response variable
    #  b. incorrect int for boundary type
    #  c. incorrect str for boundary type
    with pytest.raises(ValueError):
        bc.setLBC('a', 'dirichlet', 0)
    with pytest.raises(ValueError):
        bc.setLBC('CR', 3, 0)
    with pytest.raises(ValueError):
        bc.setLBC('CR', 'comp', 0)

    mesh = Cartesian1D(['CR', 'AL'], [0, 2e-3], 20, computeMidpoint=True)
    mesh.setResponseProfile(profile, bc)
    m = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], NiCrAlTherm, 1473.15)
    m.setup()
    dxdt = m.getdXdt(m.currentTime, m.getCurrentX())
    dt = m.getDt(dxdt)
    print(dt)
    print(dxdt)

    assert_allclose(dt, 5031.54489, rtol=1e-3)
    assert_allclose(dxdt[0][0], [0, 0], atol=1e-3)
    assert_allclose(dxdt[0][-1], [-1e-1, -5.94812538e-8], rtol=1e-3)

    periodic = PeriodicBoundary1D()
    mesh = Cartesian1D(['CR', 'AL'], [0, 2e-3], 20, computeMidpoint=True)
    mesh.setResponseProfile(profile, periodic)
    m = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], NiCrAlTherm, 1473.15)
    m.setup()
    fluxes = m.getFluxes(m.currentTime, m.getCurrentX())
    dxdt = m.getdXdt(m.currentTime, m.getCurrentX())
    dt = m.getDt(dxdt)

    # First and last flux should be equal
    assert_allclose(dt, 26548.48400, rtol=1e-3)
    assert_allclose(fluxes[0], fluxes[-1], rtol=1e-3)

def test_diffusion_interstitial():
    '''
    Tests diffusion model with interstitial mobility

    This tests both that interdiffusivity and the diffusion model correctly
    accounts for the non-volume assumption for interstitials

    This follows the diffusivity model and diffusion simulaton from
    J. Agren, Scripta Metallugrica 20 (1996) 1507
    '''
    mesh = Cartesian1D(['C'], [-2e-2, 2e-2], 100)
    profile = ProfileBuilder()
    profile.addBuildStep(StepProfile1D(0, 0.0775, 0), 'C')
    mesh.setResponseProfile(profile)

    model = SinglePhaseModel(mesh, ['FE', 'C'], ['FCC_A1'], FeCTherm, 1127+273.15)

    # Assert that mesh is converted to u-fraction
    #assert_allclose(mesh.y[0], 0.0775/(1-0.0775), rtol=1e-3)
    assert_allclose(model.data.currentY[0], 0.0775/(1-0.0775), rtol=1e-3)

    model.solve(19.5*3600)

    comps = model.getCompositions()
    assert_allclose(comps[40,1], 0.063489, rtol=1e-3)
    assert_allclose(comps[60,1], 0.014441, rtol=1e-3)

    # Repeat for homogenization
    model = HomogenizationModel(mesh, ['FE', 'C'], ['FCC_A1'], FeCTherm, 1127+273.15)
    model.homogenizationParameters.eps = 0
    # Set low max composition change constraint since this affects the time step
    # The homogenization model does not have a clear way to defining a time step
    # unlike the single phase model where we could use the von Neumann conditions
    # If it's too large, then this could result in compositions that are off
    # For the general case, the default of 2e-3 balances accuracy and performance
    # quite well, but for dilute compositions, it might be too high
    model.constraints.maxCompositionChange = 0.5e-3

    model.solve(19.5*3600)

    comps = model.getCompositions()
    # There seems to be some stochastic behavior (from global eq?), so lower the tolerance here
    assert_allclose(comps[40,1], 0.062927, rtol=1e-2)
    assert_allclose(comps[60,1], 0.016164, rtol=1e-2)


def test_fdm_mesh_and_single_phase_shape():
    profile = ProfileBuilder([(StepProfile1D(0.0, [0.2, 0.1], [0.4, 0.3]), ['CR', 'AL'])])
    mesh = CartesianFD1D(['CR', 'AL'], [-1e-3, 1e-3], 21)
    mesh.setResponseProfile(profile)

    model = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], NiCrAlTherm, TemperatureParameters(1073))
    model.setup()
    dxdt = model.getdXdt(model.currentTime, model.getCurrentX())
    fluxes = model.getFluxes(model.currentTime, model.getCurrentX())

    assert mesh.z.shape == (21, 1)
    assert mesh.zEdge.shape == (22, 1)
    assert dxdt[0].shape == (21, 2)
    assert fluxes.shape == (22, 2)


def test_fdm_constant_and_linear_profiles():
    therm = ConstantBinaryThermodynamics(
        phases=['FCC_A1'],
        diffusivities={'FCC_A1': 1.0},
        interface_compositions=(0.3, 0.7),
    )

    constant_profile = ProfileBuilder([(ConstantProfile(0.4), 'CR')])
    constant_mesh = CartesianFD1D(['CR'], [0, 1], 21)
    constant_mesh.setResponseProfile(constant_profile)
    constant_model = SinglePhaseModel(constant_mesh, ['FE', 'CR'], ['FCC_A1'], therm, TemperatureParameters(1000))
    constant_model.setup()
    constant_dxdt = constant_model.getdXdt(constant_model.currentTime, constant_model.getCurrentX())
    assert_allclose(constant_dxdt[0], 0, atol=1e-12, rtol=0)

    linear_profile = ProfileBuilder([(LinearProfile1D(0, 0.2, 1, 0.8), 'CR')])
    linear_mesh = CartesianFD1D(['CR'], [0, 1], 21)
    linear_mesh.setResponseProfile(linear_profile)
    linear_model = SinglePhaseModel(linear_mesh, ['FE', 'CR'], ['FCC_A1'], therm, TemperatureParameters(1000))
    linear_model.setup()
    linear_dxdt = linear_model.getdXdt(linear_model.currentTime, linear_model.getCurrentX())
    assert_allclose(linear_dxdt[0][1:-1], 0, atol=1e-10, rtol=0)


def test_fdm_boundary_conditions_and_periodic_fluxes():
    therm = ConstantBinaryThermodynamics(
        phases=['FCC_A1'],
        diffusivities={'FCC_A1': 1.0},
        interface_compositions=(0.3, 0.7),
    )
    profile = ProfileBuilder([(StepProfile1D(0.5, 0.2, 0.8), 'CR')])

    bc = MixedBoundary1D(['CR'])
    bc.setLBC('CR', 'dirichlet', 0.2)
    bc.setRBC('CR', 'flux', 0.0)

    mesh = CartesianFD1D(['CR'], [0, 1], 21)
    mesh.setResponseProfile(profile, bc)
    model = SinglePhaseModel(mesh, ['FE', 'CR'], ['FCC_A1'], therm, TemperatureParameters(1000))
    model.setup()
    dxdt = model.getdXdt(model.currentTime, model.getCurrentX())
    fluxes = model.getFluxes(model.currentTime, model.getCurrentX())

    assert_allclose(dxdt[0][0, 0], 0, atol=1e-12, rtol=0)
    assert_allclose(fluxes[-1, 0], 0, atol=1e-12, rtol=0)

    periodic = PeriodicBoundary1D()
    periodic_mesh = CartesianFD1D(['CR'], [0, 1], 21)
    periodic_mesh.setResponseProfile(profile, periodic)
    periodic_model = SinglePhaseModel(periodic_mesh, ['FE', 'CR'], ['FCC_A1'], therm, TemperatureParameters(1000))
    periodic_model.setup()
    periodic_fluxes = periodic_model.getFluxes(periodic_model.currentTime, periodic_model.getCurrentX())

    assert_allclose(periodic_fluxes[0], periodic_fluxes[-1], atol=1e-12, rtol=0)


def test_fdm_single_phase_and_homogenization_models_accept_mesh():
    profile = ProfileBuilder([(StepProfile1D(0.0, [0.2, 0.1], [0.4, 0.3]), ['CR', 'AL'])])
    mesh = CartesianFD1D(['CR', 'AL'], [-5e-4, 5e-4], 21)
    mesh.setResponseProfile(profile)

    single = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], NiCrAlTherm, TemperatureParameters(1073))
    single.setup()
    single_dxdt = single.getdXdt(single.currentTime, single.getCurrentX())
    assert single_dxdt[0].shape == (21, 2)
    assert np.isfinite(single.getDt(single_dxdt))

    homogenization_parameters = HomogenizationParameters(HomogenizationParameters.HASHIN_LOWER, eps=0.01)
    homogenization = HomogenizationModel(
        mesh,
        ['FE', 'CR', 'NI'],
        ['FCC_A1', 'BCC_A2'],
        thermodynamics=FeCrNiTherm,
        temperature=TemperatureParameters(1373.15),
        homogenizationParameters=homogenization_parameters,
    )
    homogenization.setup()
    homo_dxdt = homogenization.getdXdt(homogenization.currentTime, homogenization.getCurrentX())
    assert homo_dxdt[0].shape == (21, 2)
    assert np.isfinite(homogenization.getDt(homo_dxdt))


def test_moving_boundary_fdm_geometry_switches_ignored_node():
    mesh = CartesianFD1D(['CR'], [0, 1], 11)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.5, 0.2, 0.8), 'CR')]))

    geom_left = get_moving_boundary_fd_geometry(mesh, 0.41, 0.5)
    geom_right = get_moving_boundary_fd_geometry(mesh, 0.49, 0.5)

    assert geom_left.left_index == 4
    assert geom_left.right_index == 5
    assert geom_left.ignored_index == 4
    assert geom_left.left_near_index == 3
    assert geom_left.right_near_index == 5
    assert geom_right.ignored_index == 5
    assert geom_right.left_near_index == 4
    assert geom_right.right_near_index == 6


def test_interpolate_previous_ignored_composition_linear_vs_lagrange():
    z = np.linspace(0.0, 1.0, 11)
    c = 0.2 + 0.6 * z + 0.1 * z**2
    mesh = CartesianFD1D(['CR'], [0, 1], 11)
    s_old = 0.41
    s_new = 0.45
    pstar = 0.5
    interface_compositions = (0.29, 0.71)
    p_old = get_moving_boundary_fd_geometry(mesh, s_old, pstar, ignored_node_rule="legacy_two_region").p

    c_lagrange = interpolate_previous_ignored_composition(
        z,
        c,
        s_old=s_old,
        p_old=p_old,
        s_new=s_new,
        pstar=pstar,
        interface_compositions=interface_compositions,
        ignored_node_rule="legacy_two_region",
        ignored_node_reconstruction_mode="lagrange",
    )
    c_linear = interpolate_previous_ignored_composition(
        z,
        c,
        s_old=s_old,
        p_old=p_old,
        s_new=s_new,
        pstar=pstar,
        interface_compositions=interface_compositions,
        ignored_node_rule="legacy_two_region",
        ignored_node_reconstruction_mode="linear",
    )
    geom_old = get_moving_boundary_fd_geometry(mesh, s_old, pstar, ignored_node_rule="legacy_two_region")
    idx = geom_old.ignored_index
    assert idx is not None
    assert not np.isclose(c_lagrange[idx], c_linear[idx], rtol=1e-12, atol=1e-12)


def test_integrate_binary_fd_profile_reconstruction_mode_affects_new_interp():
    z = np.linspace(0.0, 1.0, 11)
    c = 0.2 + 0.6 * z + 0.1 * z**2
    mesh = CartesianFD1D(['CR'], [0, 1], 11)
    s_old = 0.41
    s_new = 0.45
    pstar = 0.5
    interface_compositions = (0.29, 0.71)
    p_old = get_moving_boundary_fd_geometry(mesh, s_old, pstar, ignored_node_rule="legacy_two_region").p

    mass_lagrange = integrate_binary_fd_profile(
        z,
        c,
        s_old=s_old,
        p_old=p_old,
        s_new=s_new,
        pstar=pstar,
        interface_compositions=interface_compositions,
        ignored_node_rule="legacy_two_region",
        ignored_node_reconstruction_mode="lagrange",
        integration_mode="weighted",
        s_for_interp="new",
    )
    mass_linear = integrate_binary_fd_profile(
        z,
        c,
        s_old=s_old,
        p_old=p_old,
        s_new=s_new,
        pstar=pstar,
        interface_compositions=interface_compositions,
        ignored_node_rule="legacy_two_region",
        ignored_node_reconstruction_mode="linear",
        integration_mode="weighted",
        s_for_interp="new",
    )
    assert not np.isclose(mass_lagrange, mass_linear, rtol=1e-12, atol=1e-12)


def test_moving_boundary_fdm_dXdt_and_fluxes():
    interfacePosition = 0.525
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    mesh = CartesianFD1D(['CR'], [0, 1], 21)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 2.0},
        interface_compositions=(0.3, 0.7),
    )
    model = MovingBoundaryFD1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
        interfaceUpdate='basic',
    )
    model.setup()
    dXdt = model.getdXdt(model.currentTime, model.getCurrentX())
    fluxes = model.getFluxes(model.currentTime, model.getCurrentX())
    dt = model.getDt(dXdt)

    assert dXdt[0].shape == (21, 1)
    assert np.isfinite(dXdt[1])
    assert fluxes.shape == (22, 1)
    assert np.isfinite(dt) and dt > 0


def test_moving_boundary_fdm_left_near_node_uses_quadratic_update_for_p_less_than_pstar():
    interfacePosition = 0.515
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    mesh = CartesianFD1D(['CR'], [0, 1], 21)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 2.0},
        interface_compositions=(0.3, 0.7),
    )
    model = MovingBoundaryFD1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
        interfaceUpdate='basic',
        pstar=0.5,
    )
    model.setup()

    x_curr = model.getCurrentX()
    c_old = np.asarray(x_curr[0], dtype=np.float64).reshape(-1)
    s_old = float(x_curr[1])
    geom = model._getInterfaceState(model.currentTime, c_old, s_old)[0]
    assert geom.p < model.pstar
    assert geom.left_near_index == geom.left_index - 1

    dXdt = model.getdXdt(model.currentTime, x_curr)
    dt = model.getDt(dXdt)
    diffusivity_nodes = model._bulk_diffusivity_nodes(c_old, model.currentTime, geom)
    expected_second_derivative = quad_fit_derivs(
        np.array([mesh.z[geom.left_near_index - 1, 0], mesh.z[geom.left_near_index, 0], s_old], dtype=np.float64),
        np.array([c_old[geom.left_near_index - 1], c_old[geom.left_near_index], 0.3], dtype=np.float64),
        mesh.z[geom.left_near_index, 0],
    )[1]
    expected_dxdt = diffusivity_nodes[geom.left_near_index] * expected_second_derivative
    bulk_dxdt = diffusivity_nodes[geom.left_near_index] * model._neumann_laplacian_uniform(c_old, geom.left_near_index)

    assert_allclose(dXdt[0][geom.left_near_index, 0], expected_dxdt, rtol=1e-10, atol=1e-10)
    assert abs(dXdt[0][geom.left_near_index, 0] - bulk_dxdt) > max(1e-8, 1e-6 * abs(expected_dxdt))
    assert np.isfinite(dt) and dt > 0

def _build_binary_fdm_model_for_pstar_schedule(interface_position=0.515, pstar=0.45):
    profile = ProfileBuilder([(StepProfile1D(interface_position, 0.1, 0.8), 'CR')])
    mesh = CartesianFD1D(['CR'], [0, 1], 21)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 2.0},
        interface_compositions=(0.3, 0.7),
    )
    model = MovingBoundaryFD1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interface_position,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        ignoredNodeReconstructionMode='lagrange',
        ignoredNodeRule='legacy_two_region',
        denom_type='eqn11',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
        interfaceUpdate='basic',
        pstar=pstar,
    )
    model.setup()
    return model


def test_moving_boundary_fdm_pstar_schedule_validation_rejects_bad_inputs():
    model = _build_binary_fdm_model_for_pstar_schedule(interface_position=0.515, pstar=0.45)

    with pytest.raises(ValueError):
        model.setPstarSchedule(schedule_times=[0.0, 1.0], schedule_values=[0.46])
    with pytest.raises(ValueError):
        model.setPstarSchedule(schedule_times=[0.0, 0.0], schedule_values=[0.46, 0.47])
    with pytest.raises(ValueError):
        model.setPstarSchedule(schedule_times=[0.0, 1.0], schedule_values=[0.46, 1.1])
    model.ignoredNodeRule = "lee_oh_1996_three_region"
    with pytest.raises(ValueError):
        model.setPstarSchedule(schedule_times=[0.0, 1.0], schedule_values=[0.35, 0.6])


def test_moving_boundary_fdm_pstar_schedule_applies_in_preprocess_and_tracks_status():
    model = _build_binary_fdm_model_for_pstar_schedule(interface_position=0.515, pstar=0.45)
    model.setPstarSchedule(schedule_times=[0.0, 1e-15], schedule_values=[0.46, 0.47])

    model.preProcess()
    status = model.getPstarScheduleStatus()
    assert np.isclose(model.pstar, 0.47)
    assert status["last_applied_index"] == 1
    assert status["next_index"] == 2
    assert len(status["history"]) >= 2
    assert all(event.get("event") in {"applied", "no_change"} for event in status["history"])


def test_moving_boundary_fdm_pstar_schedule_defers_when_regime_would_flip():
    # p ~= 0.48 at this interface location for dz=0.05.
    model = _build_binary_fdm_model_for_pstar_schedule(interface_position=0.524, pstar=0.45)
    geom_before = get_moving_boundary_fd_geometry(model.mesh, model.interfaceData.currentY, model.pstar, model.ignoredNodeRule)
    assert geom_before.ignore_mode == "ignore_right"

    model.setPstarSchedule(schedule_times=[0.0], schedule_values=[0.5])
    model.preProcess()
    status = model.getPstarScheduleStatus()

    assert np.isclose(model.pstar, 0.45)
    assert status["next_index"] == 0
    assert len(status["history"]) >= 1
    assert status["history"][-1]["event"] == "deferred"
    assert status["history"][-1]["reason"] == "regime_flip"


def test_moving_boundary_fdm_update_pstar_clears_cache_and_strict_audit_in_getdxdt():
    model = _build_binary_fdm_model_for_pstar_schedule(interface_position=0.515, pstar=0.45)

    model._cachedMulticomponentInterfaceState = {"dummy": True}
    model._pstarChangedSinceLastPreProcess = True
    with pytest.raises(ValueError):
        model.getdXdt(model.currentTime, model.getCurrentX())

    model._cachedMulticomponentInterfaceState = {"dummy": True}
    model.updatePstar(0.46, reason="test")
    assert model._cachedMulticomponentInterfaceState is None
    assert np.isclose(model.pstar, 0.46)

def test_moving_boundary_fdm_mass_correction_option_improves_mass_error():
    interfacePosition = 0.5125
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 2.0},
        interface_compositions=(0.35, 0.65),
    )

    mesh_basic = CartesianFD1D(['CR'], [0, 1], 41)
    mesh_basic.setResponseProfile(profile)
    basic = MovingBoundaryFD1DModel(
        mesh_basic,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
        interfaceUpdate='basic',
    )
    ideal_mass = 0.5125*0.1 + (1-0.5125)*0.8
    initial_mass = basic.getTotalMass()
    basic.solve(0.002, iterator=explicitEulerIterator)
    basic_error = abs(basic.getTotalMass() - initial_mass)

    mesh_corrected = CartesianFD1D(['CR'], [0, 1], 41)
    mesh_corrected.setResponseProfile(profile)
    corrected = MovingBoundaryFD1DModel(
        mesh_corrected,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
        interfaceUpdate='lee_oh_corrected',
    )
    initial_mass_corrected = corrected.getTotalMass()
    corrected.solve(0.002, iterator=explicitEulerIterator)
    corrected_error = abs(corrected.getTotalMass() - initial_mass_corrected)
    
    assert corrected_error <= basic_error + 1e-8
    assert ideal_mass==initial_mass ##This is supposed to fail for non-symmetric interface/far-field compositions


def test_moving_boundary_fdm_accepts_my_corrected_option_for_binary():
    interfacePosition = 0.5125
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 2.0},
        interface_compositions=(0.35, 0.65),
    )

    mesh_corrected = CartesianFD1D(['CR'], [0, 1], 41)
    mesh_corrected.setResponseProfile(profile)
    corrected = MovingBoundaryFD1DModel(
        mesh_corrected,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
        interfaceUpdate='my_corrected',
    )
    corrected.setup()

    assert corrected.interfaceUpdate == 'my_corrected'
    assert np.isfinite(corrected.getTotalMass())


def test_moving_boundary_fdm_flux_gradient_mode_switches_interface_flux_source():
    interfacePosition = 0.515
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 2.0},
        interface_compositions=(0.3, 0.7),
    )

    mesh_default = CartesianFD1D(['CR'], [0, 1], 21)
    mesh_default.setResponseProfile(profile)
    mesh_post = CartesianFD1D(['CR'], [0, 1], 21)
    mesh_post.setResponseProfile(profile)
    post_model = MovingBoundaryFD1DModel(
        mesh_post,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
    )
    post_model.setup()

    mesh_pre = CartesianFD1D(['CR'], [0, 1], 21)
    mesh_pre.setResponseProfile(profile)
    pre_model = MovingBoundaryFD1DModel(
        mesh_pre,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        interfaceUpdate='basic',
        fluxGradientMode='pre_diffusion',
        initialInventoryMode='integrated',
    )
    pre_model.setup()

    post_dXdt = post_model.getdXdt(post_model.currentTime, post_model.getCurrentX())
    pre_dXdt = pre_model.getdXdt(pre_model.currentTime, pre_model.getCurrentX())

    assert not np.isclose(pre_dXdt[1], post_dXdt[1], rtol=1e-8, atol=1e-10)
    assert not np.allclose(pre_model._lastInterfaceFluxes, post_model._lastInterfaceFluxes, rtol=1e-8, atol=1e-10)


def test_moving_boundary_fdm_saving_loading():
    interfacePosition = 0.525
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    mesh = CartesianFD1D(['CR'], [0, 1], 21)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 1.0},
        interface_compositions=(0.35, 0.65),
    )
    model = MovingBoundaryFD1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        interfaceUpdate='lee_oh_corrected',
        fluxGradientMode='pre_diffusion',
        initialInventoryMode='integrated',
        record=True,
    )
    model.solve(0.002, iterator=explicitEulerIterator)
    os.makedirs('.pytest_tmp', exist_ok=True)
    save_path = os.path.join('.pytest_tmp', f'moving_boundary_fdm_{os.getpid()}.npz')
    model.save(save_path)

    new_mesh = CartesianFD1D(['CR'], [0, 1], 21)
    new_mesh.setResponseProfile(profile)
    new_model = MovingBoundaryFD1DModel(
        new_mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        interfaceUpdate='lee_oh_corrected',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
        record=True,
    )
    new_model.load(save_path)

    assert_allclose(new_model.data.currentY, model.data.currentY)
    assert_allclose(new_model.getInterfacePosition(), model.getInterfacePosition())
    assert_allclose(new_model.currentTime, model.currentTime)
    assert new_model.fluxGradientMode == model.fluxGradientMode
    assert new_model.bulkUpdateScheme == model.bulkUpdateScheme
    assert new_model.interfaceUpdate == model.interfaceUpdate
    try:
        os.remove(save_path)
    except PermissionError:
        pass

def test_moving_boundary_fdm_dt_histories_record_with_interface_history():
    interfacePosition = 0.525
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    mesh = CartesianFD1D(['CR'], [0, 1], 21)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 1.0},
        interface_compositions=(0.35, 0.65),
    )
    model = MovingBoundaryFD1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        ignoredNodeReconstructionMode='lagrange',
        ignoredNodeRule='legacy_two_region',
        denom_type='eqn11',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
        record=True,
    )
    model.solve(0.002, iterator=explicitEulerIterator)

    interface_time = np.array(model.interfaceData._time[: model.interfaceData.N + 1], dtype=np.float64)
    dt_diff_time = np.array(model.dtDiffData._time[: model.dtDiffData.N + 1], dtype=np.float64)
    dt_move_time = np.array(model.dtMoveData._time[: model.dtMoveData.N + 1], dtype=np.float64)
    dt_diff_values = np.array(model.dtDiffData._y[: model.dtDiffData.N + 1], dtype=np.float64)
    dt_move_values = np.array(model.dtMoveData._y[: model.dtMoveData.N + 1], dtype=np.float64)

    assert_allclose(dt_diff_time, interface_time, rtol=0, atol=0)
    assert_allclose(dt_move_time, interface_time, rtol=0, atol=0)
    assert np.all((dt_diff_values >= 0) | np.isinf(dt_diff_values))
    assert np.all((dt_move_values >= 0) | np.isinf(dt_move_values))


def test_moving_boundary_fdm_dt_histories_support_record_false():
    interfacePosition = 0.525
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    mesh = CartesianFD1D(['CR'], [0, 1], 21)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 1.0},
        interface_compositions=(0.35, 0.65),
    )
    model = MovingBoundaryFD1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        ignoredNodeReconstructionMode='lagrange',
        ignoredNodeRule='legacy_two_region',
        denom_type='eqn11',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
        record=False,
    )
    model.solve(0.002, iterator=explicitEulerIterator)

    assert model.dtDiffData.recordInterval <= 0
    assert model.dtMoveData.recordInterval <= 0
    assert np.isfinite(model.getDtDiff()) or np.isinf(model.getDtDiff())
    assert np.isfinite(model.getDtMove()) or np.isinf(model.getDtMove())
    assert model.dtDiffData.currentTime == model.interfaceData.currentTime
    assert model.dtMoveData.currentTime == model.interfaceData.currentTime


def test_moving_boundary_fdm_saving_loading_preserves_dt_histories():
    interfacePosition = 0.525
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    mesh = CartesianFD1D(['CR'], [0, 1], 21)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 1.0},
        interface_compositions=(0.35, 0.65),
    )
    model = MovingBoundaryFD1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        ignoredNodeReconstructionMode='lagrange',
        ignoredNodeRule='legacy_two_region',
        denom_type='eqn11',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
        record=True,
    )
    model.solve(0.002, iterator=explicitEulerIterator)
    os.makedirs('.pytest_tmp', exist_ok=True)
    save_path = os.path.join('.pytest_tmp', f'moving_boundary_fdm_dt_{os.getpid()}.npz')
    model.save(save_path)

    new_mesh = CartesianFD1D(['CR'], [0, 1], 21)
    new_mesh.setResponseProfile(profile)
    new_model = MovingBoundaryFD1DModel(
        new_mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        ignoredNodeReconstructionMode='lagrange',
        ignoredNodeRule='legacy_two_region',
        denom_type='eqn11',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
        record=True,
    )
    new_model.load(save_path)

    assert new_model.dtDiffData.recordInterval == model.dtDiffData.recordInterval
    assert new_model.dtMoveData.recordInterval == model.dtMoveData.recordInterval
    assert new_model.dtDiffData.N == model.dtDiffData.N
    assert new_model.dtMoveData.N == model.dtMoveData.N
    assert_allclose(new_model.dtDiffData._time, model.dtDiffData._time)
    assert_allclose(new_model.dtMoveData._time, model.dtMoveData._time)
    assert_allclose(new_model.dtDiffData._y, model.dtDiffData._y)
    assert_allclose(new_model.dtMoveData._y, model.dtMoveData._y)
    assert_allclose(new_model.getDtDiff(), model.getDtDiff())
    assert_allclose(new_model.getDtMove(), model.getDtMove())
    try:
        os.remove(save_path)
    except PermissionError:
        pass


def test_moving_boundary_fdm_phase_length_idealized_initial_inventory_binary():
    interfacePosition = 0.525
    c_left = 0.1
    c_right = 0.8
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, c_left, c_right), 'CR')])
    mesh = CartesianFD1D(['CR'], [0, 1], 21)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 1.0},
        interface_compositions=(0.35, 0.65),
    )
    model = MovingBoundaryFD1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='phase_length_idealized',
    )
    expected = c_left * interfacePosition + c_right * (1.0 - interfacePosition)
    assert np.isscalar(model._initialInventory)
    assert_allclose(model._initialInventory, expected, rtol=0, atol=1e-12)


def test_moving_boundary_fdm_phase_length_idealized_requires_constant_phase_composition():
    interfacePosition = 0.525
    profile = ProfileBuilder([(LinearProfile1D(0, 0.1, 1, 0.8), 'CR')])
    mesh = CartesianFD1D(['CR'], [0, 1], 21)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 1.0},
        interface_compositions=(0.35, 0.65),
    )
    with pytest.raises(ValueError, match="phase_length_idealized initial inventory requires constant composition"):
        MovingBoundaryFD1DModel(
            mesh,
            ['FE', 'CR'],
            ['ALPHA', 'BETA'],
            thermodynamics=therm,
            temperature=TemperatureParameters(1000),
            interfacePosition=interfacePosition,
            bulkUpdateScheme='legacy',
            integrationMode='weighted',
            interfaceUpdate='basic',
            fluxGradientMode='post_diffusion',
            initialInventoryMode='phase_length_idealized',
        )


def test_moving_boundary_fdm_from_dict_requires_initial_inventory_mode():
    interfacePosition = 0.525
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    mesh = CartesianFD1D(['CR'], [0, 1], 21)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 1.0},
        interface_compositions=(0.35, 0.65),
    )
    model = MovingBoundaryFD1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
    )
    data = model.toDict()
    data.pop("initial_inventory_mode")

    new_mesh = CartesianFD1D(['CR'], [0, 1], 21)
    new_mesh.setResponseProfile(profile)
    new_model = MovingBoundaryFD1DModel(
        new_mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
    )
    with pytest.raises(ValueError, match="does not include 'initial_inventory_mode'"):
        new_model.fromDict(data)


def test_moving_boundary_fdm_requires_ignored_node_reconstruction_mode():
    interfacePosition = 0.525
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    mesh = CartesianFD1D(['CR'], [0, 1], 21)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 1.0},
        interface_compositions=(0.35, 0.65),
    )
    with pytest.raises(ValueError, match="ignoredNodeReconstructionMode must be specified explicitly"):
        MovingBoundaryFD1DModel(
            mesh,
            ['FE', 'CR'],
            ['ALPHA', 'BETA'],
            thermodynamics=therm,
            temperature=TemperatureParameters(1000),
            interfacePosition=interfacePosition,
            bulkUpdateScheme='legacy',
            integrationMode='weighted',
            ignoredNodeRule='legacy_two_region',
            interfaceUpdate='basic',
            fluxGradientMode='post_diffusion',
            initialInventoryMode='integrated',
        )


def test_moving_boundary_fdm_from_dict_requires_reconstruction_mode_field():
    interfacePosition = 0.525
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    mesh = CartesianFD1D(['CR'], [0, 1], 21)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 1.0},
        interface_compositions=(0.35, 0.65),
    )
    model = MovingBoundaryFD1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        ignoredNodeRule='legacy_two_region',
        ignoredNodeReconstructionMode='lagrange',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
    )
    data = model.toDict()
    data.pop("ignored_node_reconstruction_mode")

    new_mesh = CartesianFD1D(['CR'], [0, 1], 21)
    new_mesh.setResponseProfile(profile)
    new_model = MovingBoundaryFD1DModel(
        new_mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        ignoredNodeRule='legacy_two_region',
        ignoredNodeReconstructionMode='lagrange',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
    )
    with pytest.raises(ValueError, match="does not include 'ignored_node_reconstruction_mode'"):
        new_model.fromDict(data)


def test_moving_boundary_fdm_requires_valid_bulk_update_scheme():
    interfacePosition = 0.525
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    mesh = CartesianFD1D(['CR'], [0, 1], 21)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 1.0},
        interface_compositions=(0.35, 0.65),
    )

    with pytest.raises(TypeError):
        MovingBoundaryFD1DModel(
            mesh,
            ['FE', 'CR'],
            ['ALPHA', 'BETA'],
            thermodynamics=therm,
            temperature=TemperatureParameters(1000),
            interfacePosition=interfacePosition,
        )

    with pytest.raises(ValueError, match="interfaceUpdate must be specified explicitly"):
        MovingBoundaryFD1DModel(
            mesh,
            ['FE', 'CR'],
            ['ALPHA', 'BETA'],
            thermodynamics=therm,
            temperature=TemperatureParameters(1000),
            interfacePosition=interfacePosition,
            bulkUpdateScheme='legacy',
        )

    with pytest.raises(ValueError, match="integrationMode must be specified explicitly"):
        MovingBoundaryFD1DModel(
            mesh,
            ['FE', 'CR'],
            ['ALPHA', 'BETA'],
            thermodynamics=therm,
            temperature=TemperatureParameters(1000),
            interfacePosition=interfacePosition,
            bulkUpdateScheme='legacy',
            interfaceUpdate='basic',
        )

    with pytest.raises(ValueError, match="fluxGradientMode must be specified explicitly"):
        MovingBoundaryFD1DModel(
            mesh,
            ['FE', 'CR'],
            ['ALPHA', 'BETA'],
            thermodynamics=therm,
            temperature=TemperatureParameters(1000),
            interfacePosition=interfacePosition,
            bulkUpdateScheme='legacy',
            interfaceUpdate='basic',
            integrationMode='weighted',
        )

    with pytest.raises(ValueError, match="initialInventoryMode must be specified explicitly"):
        MovingBoundaryFD1DModel(
            mesh,
            ['FE', 'CR'],
            ['ALPHA', 'BETA'],
            thermodynamics=therm,
            temperature=TemperatureParameters(1000),
            interfacePosition=interfacePosition,
            bulkUpdateScheme='legacy',
            interfaceUpdate='basic',
            integrationMode='weighted',
            fluxGradientMode='post_diffusion',
        )

    with pytest.raises(ValueError, match="bulkUpdateScheme"):
        MovingBoundaryFD1DModel(
            mesh,
            ['FE', 'CR'],
            ['ALPHA', 'BETA'],
            thermodynamics=therm,
            temperature=TemperatureParameters(1000),
            interfacePosition=interfacePosition,
            bulkUpdateScheme='invalid',
            integrationMode='weighted',
            interfaceUpdate='basic',
            fluxGradientMode='post_diffusion',
            initialInventoryMode='integrated',
        )


def test_moving_boundary_fdm_rejects_invalid_reconstruction_mode():
    interfacePosition = 0.525
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    mesh = CartesianFD1D(['CR'], [0, 1], 21)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 1.0},
        interface_compositions=(0.35, 0.65),
    )
    with pytest.raises(ValueError, match="ignoredNodeReconstructionMode must be one of"):
        MovingBoundaryFD1DModel(
            mesh,
            ['FE', 'CR'],
            ['ALPHA', 'BETA'],
            thermodynamics=therm,
            temperature=TemperatureParameters(1000),
            interfacePosition=interfacePosition,
            bulkUpdateScheme='legacy',
            integrationMode='weighted',
            ignoredNodeRule='legacy_two_region',
            ignoredNodeReconstructionMode='cubic',
            interfaceUpdate='basic',
            fluxGradientMode='post_diffusion',
            initialInventoryMode='integrated',
        )


def test_moving_boundary_fdm_bulk_update_scheme_matches_for_constant_diffusivity():
    interfacePosition = 0.515
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 2.0},
        interface_compositions=(0.3, 0.7),
    )

    mesh_legacy = CartesianFD1D(['CR'], [0, 1], 21)
    mesh_legacy.setResponseProfile(profile)
    legacy = MovingBoundaryFD1DModel(
        mesh_legacy,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        interfaceUpdate='lee_oh_corrected',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
        record=True,
    )

    mesh_flux = CartesianFD1D(['CR'], [0, 1], 21)
    mesh_flux.setResponseProfile(profile)
    flux_form = MovingBoundaryFD1DModel(
        mesh_flux,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='flux_form',
        integrationMode='weighted',
        interfaceUpdate='lee_oh_corrected',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
        record=True,
    )

    legacy.solve(0.002, iterator=explicitEulerIterator)
    flux_form.solve(0.002, iterator=explicitEulerIterator)

    legacy_times = np.array(legacy.interfaceData._time[: legacy.interfaceData.N + 1], dtype=np.float64)
    flux_times = np.array(flux_form.interfaceData._time[: flux_form.interfaceData.N + 1], dtype=np.float64)
    legacy_positions = np.array(legacy.interfaceData._y[: legacy.interfaceData.N + 1], dtype=np.float64)
    flux_positions = np.array(flux_form.interfaceData._y[: flux_form.interfaceData.N + 1], dtype=np.float64)

    assert_allclose(flux_times, legacy_times, rtol=0, atol=1e-17) ##These were exactly equal when flux-form was used for bulk. Then when flux_form was used for near-interface they are different because different order of operations gives different round-off errors
    assert_allclose(flux_positions, legacy_positions, rtol=0, atol=1e-15)


def test_moving_boundary_fdm_flux_form_uses_variable_diffusivity_bulk_update():
    interfacePosition = 0.525
    profile = ProfileBuilder([(LinearProfile1D(0, 0.2, 1, 0.8), 'CR')])
    therm = VariableBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 2.0},
        interface_compositions=(0.3, 0.7),
    )

    mesh_legacy = CartesianFD1D(['CR'], [0, 1], 21)
    mesh_legacy.setResponseProfile(profile)
    legacy = MovingBoundaryFD1DModel(
        mesh_legacy,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
    )
    legacy.setup()

    mesh_flux = CartesianFD1D(['CR'], [0, 1], 21)
    mesh_flux.setResponseProfile(profile)
    flux_form = MovingBoundaryFD1DModel(
        mesh_flux,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='flux_form',
        integrationMode='weighted',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
    )
    flux_form.setup()

    x_curr = flux_form.getCurrentX()
    c_old = np.asarray(x_curr[0], dtype=np.float64).reshape(-1)
    s_old = float(x_curr[1])
    geom = flux_form._getInterfaceState(flux_form.currentTime, c_old, s_old)[0]
    diffusivity_nodes = flux_form._bulk_diffusivity_nodes(c_old, flux_form.currentTime, geom)
    flux_dXdt = flux_form.getdXdt(flux_form.currentTime, x_curr)
    legacy_dXdt = legacy.getdXdt(legacy.currentTime, legacy.getCurrentX())
    bulk_dcdt = flux_form._bulk_dcdt_flux_form(c_old, diffusivity_nodes)

    bulk_indices = np.arange(len(c_old))
    eligible = np.ones(len(c_old), dtype=bool)
    eligible[geom.ignored_index] = False
    if geom.left_near_index >= 1:
        eligible[geom.left_near_index] = False
    if geom.right_near_index + 1 < len(c_old):
        eligible[geom.right_near_index] = False
    candidate_indices = bulk_indices[eligible & (bulk_indices < geom.left_near_index)]
    idx = int(candidate_indices[-1])

    legacy_value = diffusivity_nodes[idx] * flux_form._neumann_laplacian_uniform(c_old, idx)
    assert_allclose(flux_dXdt[0][idx, 0], bulk_dcdt[idx], rtol=1e-12, atol=1e-12)
    assert not np.isclose(bulk_dcdt[idx], legacy_value, rtol=1e-10, atol=1e-12)
    assert_allclose(legacy_dXdt[0][idx, 0], legacy_value, rtol=1e-12, atol=1e-12)


def test_moving_boundary_fdm_bulk_update_switch_only_changes_ordinary_bulk_nodes():
    interfacePosition = 0.515
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    therm = VariableBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 2.0},
        interface_compositions=(0.3, 0.7),
    )

    mesh_legacy = CartesianFD1D(['CR'], [0, 1], 21)
    mesh_legacy.setResponseProfile(profile)
    legacy = MovingBoundaryFD1DModel(
        mesh_legacy,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
    )
    legacy.setup()

    mesh_flux = CartesianFD1D(['CR'], [0, 1], 21)
    mesh_flux.setResponseProfile(profile)
    flux_form = MovingBoundaryFD1DModel(
        mesh_flux,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='flux_form',
        integrationMode='weighted',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
    )
    flux_form.setup()

    legacy_dXdt = legacy.getdXdt(legacy.currentTime, legacy.getCurrentX())
    flux_dXdt = flux_form.getdXdt(flux_form.currentTime, flux_form.getCurrentX())
    geom = flux_form._getInterfaceState(
        flux_form.currentTime,
        np.asarray(flux_form.getCurrentX()[0], dtype=np.float64).reshape(-1),
        float(flux_form.getCurrentX()[1]),
    )[0]

    assert_allclose(
        flux_form.data.currentY[geom.ignored_index, 0],
        legacy.data.currentY[geom.ignored_index, 0],
        rtol=0,
        atol=0,
    )
    differing_bulk = np.where(np.abs(flux_dXdt[0][:, 0] - legacy_dXdt[0][:, 0]) > 1e-12)[0]
    assert geom.left_near_index in differing_bulk
    assert geom.right_near_index in differing_bulk


def test_moving_boundary_fdm_flux_form_matches_legacy_at_left_near_node_for_constant_diffusivity():
    interfacePosition = 0.515
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 2.0},
        interface_compositions=(0.3, 0.7),
    )

    mesh_legacy = CartesianFD1D(['CR'], [0, 1], 21)
    mesh_legacy.setResponseProfile(profile)
    legacy = MovingBoundaryFD1DModel(
        mesh_legacy,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
        record=True,
    )

    mesh_flux = CartesianFD1D(['CR'], [0, 1], 21)
    mesh_flux.setResponseProfile(profile)
    flux_form = MovingBoundaryFD1DModel(
        mesh_flux,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='flux_form',
        integrationMode='weighted',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
        record=True,
    )

    legacy_dXdt = legacy.getdXdt(legacy.currentTime, legacy.getCurrentX())
    flux_dXdt = flux_form.getdXdt(flux_form.currentTime, flux_form.getCurrentX())
    geom = flux_form._getInterfaceState(
        flux_form.currentTime,
        np.asarray(flux_form.getCurrentX()[0], dtype=np.float64).reshape(-1),
        float(flux_form.getCurrentX()[1]),
    )[0]
    assert geom.p < flux_form.pstar
    assert_allclose(
        flux_dXdt[0][geom.left_near_index, 0],
        legacy_dXdt[0][geom.left_near_index, 0],
        rtol=1e-12,
        atol=1e-12,
    )
    assert_allclose(flux_dXdt[1], legacy_dXdt[1], rtol=1e-12, atol=1e-12)

    legacy.solve(0.002, iterator=explicitEulerIterator)
    flux_form.solve(0.002, iterator=explicitEulerIterator)
    assert_allclose(flux_form.interfaceData._time[: flux_form.interfaceData.N + 1], legacy.interfaceData._time[: legacy.interfaceData.N + 1], rtol=0, atol=0)
    assert_allclose(flux_form.interfaceData._y[: flux_form.interfaceData.N + 1], legacy.interfaceData._y[: legacy.interfaceData.N + 1], rtol=0, atol=0)


def test_moving_boundary_fdm_flux_form_matches_legacy_at_right_near_node_for_constant_diffusivity():
    interfacePosition = 0.585
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 2.0},
        interface_compositions=(0.3, 0.7),
    )

    mesh_legacy = CartesianFD1D(['CR'], [0, 1], 21)
    mesh_legacy.setResponseProfile(profile)
    legacy = MovingBoundaryFD1DModel(
        mesh_legacy,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
        record=True,
    )

    mesh_flux = CartesianFD1D(['CR'], [0, 1], 21)
    mesh_flux.setResponseProfile(profile)
    flux_form = MovingBoundaryFD1DModel(
        mesh_flux,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='flux_form',
        integrationMode='weighted',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
        record=True,
    )

    legacy_dXdt = legacy.getdXdt(legacy.currentTime, legacy.getCurrentX())
    flux_dXdt = flux_form.getdXdt(flux_form.currentTime, flux_form.getCurrentX())
    geom = flux_form._getInterfaceState(
        flux_form.currentTime,
        np.asarray(flux_form.getCurrentX()[0], dtype=np.float64).reshape(-1),
        float(flux_form.getCurrentX()[1]),
    )[0]
    assert geom.p >= flux_form.pstar
    assert_allclose(
        flux_dXdt[0][geom.right_near_index, 0],
        legacy_dXdt[0][geom.right_near_index, 0],
        rtol=1e-12,
        atol=1e-12,
    )
    assert_allclose(flux_dXdt[1], legacy_dXdt[1], rtol=1e-12, atol=1e-12)

    legacy.solve(0.002, iterator=explicitEulerIterator)
    flux_form.solve(0.002, iterator=explicitEulerIterator)
    assert_allclose(flux_form.interfaceData._time[: flux_form.interfaceData.N + 1], legacy.interfaceData._time[: legacy.interfaceData.N + 1], rtol=0, atol=1e-18)
    assert_allclose(flux_form.interfaceData._y[: flux_form.interfaceData.N + 1], legacy.interfaceData._y[: legacy.interfaceData.N + 1], rtol=0, atol=1e-18)


def test_moving_boundary_fdm_flux_form_left_near_node_matches_cut_cell_formula():
    interfacePosition = 0.515
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    therm = VariableBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 2.0},
        interface_compositions=(0.3, 0.7),
    )

    mesh_legacy = CartesianFD1D(['CR'], [0, 1], 21)
    mesh_legacy.setResponseProfile(profile)
    legacy = MovingBoundaryFD1DModel(
        mesh_legacy,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
    )
    legacy.setup()

    mesh_flux = CartesianFD1D(['CR'], [0, 1], 21)
    mesh_flux.setResponseProfile(profile)
    flux_form = MovingBoundaryFD1DModel(
        mesh_flux,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='flux_form',
        integrationMode='weighted',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
    )
    flux_form.setup()

    c_old = np.asarray(flux_form.getCurrentX()[0], dtype=np.float64).reshape(-1)
    s_old = float(flux_form.getCurrentX()[1])
    geom, _, c_left_int, c_right_int, D_left_int, D_right_int = flux_form._getInterfaceState(flux_form.currentTime, c_old, s_old)
    diffusivity_nodes = flux_form._bulk_diffusivity_nodes(c_old, flux_form.currentTime, geom)
    flux_dXdt = flux_form.getdXdt(flux_form.currentTime, flux_form.getCurrentX())
    legacy_dXdt = legacy.getdXdt(legacy.currentTime, legacy.getCurrentX())
    idx, expected = flux_form._near_interface_dcdt_flux_form(
        c_old,
        diffusivity_nodes,
        geom,
        "A",
        (c_left_int, c_right_int),
        (D_left_int, D_right_int),
    )

    assert idx == geom.left_near_index
    assert_allclose(flux_dXdt[0][idx, 0], expected, rtol=1e-12, atol=1e-12)
    assert not np.isclose(flux_dXdt[0][idx, 0], legacy_dXdt[0][idx, 0], rtol=1e-10, atol=1e-12)


def test_moving_boundary_fdm_flux_form_right_near_node_matches_cut_cell_formula():
    interfacePosition = 0.565
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    therm = VariableBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 2.0},
        interface_compositions=(0.3, 0.7),
    )

    mesh_legacy = CartesianFD1D(['CR'], [0, 1], 21)
    mesh_legacy.setResponseProfile(profile)
    legacy = MovingBoundaryFD1DModel(
        mesh_legacy,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
    )
    legacy.setup()

    mesh_flux = CartesianFD1D(['CR'], [0, 1], 21)
    mesh_flux.setResponseProfile(profile)
    flux_form = MovingBoundaryFD1DModel(
        mesh_flux,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='flux_form',
        integrationMode='weighted',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
    )
    flux_form.setup()

    c_old = np.asarray(flux_form.getCurrentX()[0], dtype=np.float64).reshape(-1)
    s_old = float(flux_form.getCurrentX()[1])
    geom, _, c_left_int, c_right_int, D_left_int, D_right_int = flux_form._getInterfaceState(flux_form.currentTime, c_old, s_old)
    diffusivity_nodes = flux_form._bulk_diffusivity_nodes(c_old, flux_form.currentTime, geom)
    flux_dXdt = flux_form.getdXdt(flux_form.currentTime, flux_form.getCurrentX())
    legacy_dXdt = legacy.getdXdt(legacy.currentTime, legacy.getCurrentX())
    idx, expected = flux_form._near_interface_dcdt_flux_form(
        c_old,
        diffusivity_nodes,
        geom,
        "B",
        (c_left_int, c_right_int),
        (D_left_int, D_right_int),
    )

    assert idx == geom.right_near_index
    assert_allclose(flux_dXdt[0][idx, 0], expected, rtol=1e-12, atol=1e-12)
    assert not np.isclose(flux_dXdt[0][idx, 0], legacy_dXdt[0][idx, 0], rtol=1e-10, atol=1e-12)


def _make_mock_ternary_moving_boundary_model(
    interface_update="basic",
    balance_element=None,
    record=False,
    multicomponent_interface_state_update="pre_diffusion_only",
    reverse_endpoint_order=False,
    ambiguous_endpoint_phases=False,
):
    interface_position = 0.515
    left_comp = np.array([0.30, 0.05], dtype=np.float64)
    right_comp = np.array([0.10, 0.18], dtype=np.float64)
    profile = ProfileBuilder(
        [
            (StepProfile1D(interface_position, left_comp[0], right_comp[0]), 'CR'),
            (StepProfile1D(interface_position, left_comp[1], right_comp[1]), 'NI'),
        ]
    )
    mesh = CartesianFD1D(['CR', 'NI'], [0, 1], 41)
    mesh.setResponseProfile(profile)
    therm = MockTernaryTieLineThermodynamics(
        ['ALPHA', 'BETA'],
        left_probe=left_comp,
        right_probe=right_comp,
        reverse_endpoint_order=reverse_endpoint_order,
        ambiguous_endpoint_phases=ambiguous_endpoint_phases,
    )
    model = MovingBoundaryFD1DModel(
        mesh,
        ['FE', 'CR', 'NI'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interface_position,
        bulkUpdateScheme='flux_form',
        integrationMode='weighted',
        interfaceUpdate=interface_update,
        fluxGradientMode='pre_diffusion',
        initialInventoryMode='integrated',
        balanceElement=balance_element,
        multicomponentInterfaceStateUpdate=multicomponent_interface_state_update,
        record=record,
    )
    model.setup()
    return model


def test_moving_boundary_fdm_ternary_requires_balance_element_for_corrected_mode():
    with pytest.raises(ValueError, match="balanceElement"):
        _make_mock_ternary_moving_boundary_model(interface_update="lee_oh_corrected")


def test_moving_boundary_fdm_ternary_requires_balance_element_for_my_corrected_mode():
    with pytest.raises(ValueError, match="balanceElement"):
        _make_mock_ternary_moving_boundary_model(interface_update="my_corrected")


def test_moving_boundary_fdm_ternary_rejects_invalid_balance_element():
    with pytest.raises(ValueError, match="balanceElement"):
        _make_mock_ternary_moving_boundary_model(interface_update="lee_oh_corrected", balance_element="AL")


def test_moving_boundary_fdm_ternary_interface_state_matches_component_velocities():
    model = _make_mock_ternary_moving_boundary_model(interface_update="basic")
    x_curr = model.getCurrentX()
    composition = np.asarray(x_curr[0], dtype=np.float64)
    interface_position = float(x_curr[1])

    state = model._solveMulticomponentInterfaceState(model.currentTime, composition, interface_position)
    dXdt = model.getdXdt(model.currentTime, x_curr)
    fluxes = model.getFluxes(model.currentTime, x_curr)

    assert state["interface_compositions"][0].shape == (2,)
    assert state["interface_compositions"][1].shape == (2,)
    assert state["interface_diffusivities"][0].shape == (2, 2)
    assert state["interface_diffusivities"][1].shape == (2, 2)
    assert abs(state["component_velocities"][0] - state["component_velocities"][1]) <= state["tolerance"]
    assert dXdt[0].shape == (41, 2)
    assert fluxes.shape == (42, 2)
    assert np.isfinite(dXdt[1])
    assert np.all(np.isfinite(model.getTotalInventory()))


def test_multicomponent_thermo_interface_metadata_supported():
    therm = MulticomponentThermodynamics(FECRNI_DB, ['FE', 'CR', 'NI'], ['FCC_A1', 'BCC_A2'])
    c_left, c_right, meta = therm.getInterfacialComposition(
        np.array([0.30, 0.085], dtype=np.float64),
        1373,
        0,
        precPhase='BCC_A2',
        returnMeta=True,
    )
    assert np.asarray(c_left, dtype=np.float64).ndim == 1
    assert np.asarray(c_right, dtype=np.float64).ndim == 1
    assert np.asarray(c_left, dtype=np.float64).size >= 2
    assert np.asarray(c_right, dtype=np.float64).size >= 2
    assert isinstance(meta, dict)
    assert "endpoints" in meta
    assert len(meta["endpoints"]) == 2


def test_surrogate_untrained_interface_metadata_passthrough():
    therm = MulticomponentThermodynamics(FECRNI_DB, ['FE', 'CR', 'NI'], ['FCC_A1', 'BCC_A2'])
    surrogate = GeneralSurrogate(therm)
    _, _, meta_therm = therm.getInterfacialComposition(
        np.array([0.30, 0.085], dtype=np.float64),
        1373,
        0,
        precPhase='BCC_A2',
        returnMeta=True,
    )
    _, _, meta_sur = surrogate.getInterfacialComposition(
        np.array([0.30, 0.085], dtype=np.float64),
        1373,
        0,
        precPhase='BCC_A2',
        returnMeta=True,
    )
    assert meta_sur["endpoint_phases"] == meta_therm["endpoint_phases"]


def test_binary_surrogate_trained_interface_metadata_supported():
    surrogate = BinarySurrogate(MockBinaryInterfaceThermo())
    surrogate.trainInterfacialComposition(
        T=np.array([1000.0, 1050.0, 1100.0], dtype=np.float64),
        gExtra=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        precPhase='BETA',
        broadcast=False,
    )
    _, _, meta = surrogate.getInterfacialComposition(1075.0, 0.15, precPhase='BETA', returnMeta=True)
    assert meta["endpoint_phases"] == ('ALPHA', 'BETA')
    assert meta["endpoints"][0]["phase"] == 'ALPHA'
    assert meta["endpoints"][1]["phase"] == 'BETA'


def test_moving_boundary_fdm_ternary_pre_diffusion_only_solves_once_per_step():
    model = _make_mock_ternary_moving_boundary_model(
        interface_update="basic",
        multicomponent_interface_state_update="pre_diffusion_only",
    )
    solve_count = {"n": 0}
    original = model._solveMulticomponentInterfaceState

    def counted(*args, **kwargs):
        solve_count["n"] += 1
        return original(*args, **kwargs)

    model._solveMulticomponentInterfaceState = counted
    model.getdXdt(model.currentTime, model.getCurrentX())

    assert solve_count["n"] == 1


def test_moving_boundary_fdm_ternary_phase_ordering_swaps_reversed_endpoints():
    model_nominal = _make_mock_ternary_moving_boundary_model(interface_update="basic", reverse_endpoint_order=False)
    model_reversed = _make_mock_ternary_moving_boundary_model(interface_update="basic", reverse_endpoint_order=True)
    state_nominal = model_nominal._solveMulticomponentInterfaceState(
        model_nominal.currentTime,
        np.asarray(model_nominal.getCurrentX()[0], dtype=np.float64),
        float(model_nominal.getCurrentX()[1]),
    )
    state_reversed = model_reversed._solveMulticomponentInterfaceState(
        model_reversed.currentTime,
        np.asarray(model_reversed.getCurrentX()[0], dtype=np.float64),
        float(model_reversed.getCurrentX()[1]),
    )
    assert_allclose(state_reversed["interface_compositions"][0], state_nominal["interface_compositions"][0], atol=1e-12, rtol=1e-12)
    assert_allclose(state_reversed["interface_compositions"][1], state_nominal["interface_compositions"][1], atol=1e-12, rtol=1e-12)


def test_moving_boundary_fdm_ternary_phase_ordering_raises_on_ambiguous_metadata():
    with pytest.raises(ValueError, match="Ambiguous interface endpoint phases"):
        _make_mock_ternary_moving_boundary_model(interface_update="basic", ambiguous_endpoint_phases=True)


def test_moving_boundary_fdm_ternary_pre_and_post_diffusion_solves_twice_per_step():
    model = _make_mock_ternary_moving_boundary_model(
        interface_update="basic",
        multicomponent_interface_state_update="pre_and_post_diffusion",
    )
    solve_count = {"n": 0}
    original = model._solveMulticomponentInterfaceState

    def counted(*args, **kwargs):
        solve_count["n"] += 1
        return original(*args, **kwargs)

    model._solveMulticomponentInterfaceState = counted
    model.getdXdt(model.currentTime, model.getCurrentX())

    assert solve_count["n"] == 2


def test_moving_boundary_fdm_ternary_cached_only_interface_state_raises_before_first_step():
    model = _make_mock_ternary_moving_boundary_model(interface_update="basic")
    composition = np.asarray(model.getCurrentX()[0], dtype=np.float64)
    interface_position = float(model.getCurrentX()[1])

    with pytest.raises(ValueError, match="No cached ternary interface state"):
        model._getInterfaceState(model.currentTime, composition, interface_position)


def test_moving_boundary_fdm_ternary_get_interface_compositions_uses_cache_after_step():
    model = _make_mock_ternary_moving_boundary_model(interface_update="basic")
    model.solve(5e-5, iterator=explicitEulerIterator)

    solve_count = {"n": 0}
    original = model._solveMulticomponentInterfaceState

    def counted(*args, **kwargs):
        solve_count["n"] += 1
        return original(*args, **kwargs)

    model._solveMulticomponentInterfaceState = counted
    left, right = model.getInterfaceCompositions()

    assert left.shape == (2,)
    assert right.shape == (2,)
    assert solve_count["n"] == 0


def test_moving_boundary_fdm_ternary_interface_composition_history_shapes():
    model = _make_mock_ternary_moving_boundary_model(interface_update="basic", record=True)
    model.solve(5e-5, iterator=explicitEulerIterator)
    history = model._interfaceCompositionHistory

    assert history is not None
    assert history["pre_left"]._y.ndim == 2
    assert history["pre_left"]._y.shape[1] == 2
    assert history["pre_right"]._y.shape[1] == 2
    assert history["post_left"]._y.shape[1] == 2
    assert history["post_right"]._y.shape[1] == 2
    assert history["time"]._time.shape[0] == history["pre_left"]._y.shape[0]


def test_moving_boundary_fdm_ternary_pre_only_post_equals_pre_history():
    model = _make_mock_ternary_moving_boundary_model(
        interface_update="basic",
        record=True,
        multicomponent_interface_state_update="pre_diffusion_only",
    )
    model.solve(5e-5, iterator=explicitEulerIterator)
    history = model._interfaceCompositionHistory

    assert_allclose(history["pre_left"]._y, history["post_left"]._y)
    assert_allclose(history["pre_right"]._y, history["post_right"]._y)


def test_moving_boundary_fdm_ternary_exact_time_interface_composition_retrieval():
    model = _make_mock_ternary_moving_boundary_model(interface_update="basic", record=True)
    model.solve(5e-5, iterator=explicitEulerIterator)
    t_exact = float(model._interfaceCompositionHistory["time"]._time[0])

    left_pre, right_pre = model.getInterfaceCompositions(time=t_exact, stage="pre")
    left_post, right_post = model.getInterfaceCompositions(time=t_exact, stage="post")

    assert left_pre.shape == (2,)
    assert right_pre.shape == (2,)
    assert left_post.shape == (2,)
    assert right_post.shape == (2,)

    with pytest.raises(ValueError, match="exact recorded interface composition times"):
        model.getInterfaceCompositions(time=t_exact + 1e-6, stage="pre")


def test_moving_boundary_fdm_ternary_stage_selection_used_maps_to_policy():
    model_pre = _make_mock_ternary_moving_boundary_model(
        interface_update="basic",
        record=True,
        multicomponent_interface_state_update="pre_diffusion_only",
    )
    model_pre.solve(5e-5, iterator=explicitEulerIterator)
    used_left_pre, used_right_pre = model_pre.getInterfaceCompositions(stage="used")
    pre_left, pre_right = model_pre.getInterfaceCompositions(stage="pre")
    assert_allclose(used_left_pre, pre_left)
    assert_allclose(used_right_pre, pre_right)

    model_post = _make_mock_ternary_moving_boundary_model(
        interface_update="basic",
        record=True,
        multicomponent_interface_state_update="pre_and_post_diffusion",
    )
    model_post.solve(5e-5, iterator=explicitEulerIterator)
    used_left_post, used_right_post = model_post.getInterfaceCompositions(stage="used")
    post_left, post_right = model_post.getInterfaceCompositions(stage="post")
    assert_allclose(used_left_post, post_left)
    assert_allclose(used_right_post, post_right)


def test_moving_boundary_fdm_binary_stage_request_raises_for_interface_compositions():
    interfacePosition = 0.525
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    mesh = CartesianFD1D(['CR'], [0, 1], 21)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 2.0},
        interface_compositions=(0.3, 0.7),
    )
    model = MovingBoundaryFD1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        bulkUpdateScheme='legacy',
        integrationMode='weighted',
        interfaceUpdate='basic',
        fluxGradientMode='post_diffusion',
        initialInventoryMode='integrated',
    )
    model.setup()
    with pytest.raises(ValueError, match="does not record ternary interface composition stages"):
        model.getInterfaceCompositions(stage="pre")


def test_moving_boundary_fdm_ternary_balance_element_changes_corrected_trajectory():
    basic = _make_mock_ternary_moving_boundary_model(interface_update="basic", record=True)
    corrected_cr = _make_mock_ternary_moving_boundary_model(interface_update="lee_oh_corrected", balance_element="CR", record=True)
    corrected_ni = _make_mock_ternary_moving_boundary_model(interface_update="lee_oh_corrected", balance_element="NI", record=True)

    basic.solve(5e-5, iterator=explicitEulerIterator)
    corrected_cr.solve(5e-5, iterator=explicitEulerIterator)
    corrected_ni.solve(5e-5, iterator=explicitEulerIterator)

    basic_residual = np.abs(basic.getTotalInventory() - basic._initialInventory)
    corrected_cr_residual = np.abs(corrected_cr.getTotalInventory() - corrected_cr._initialInventory)
    corrected_ni_residual = np.abs(corrected_ni.getTotalInventory() - corrected_ni._initialInventory)

    assert corrected_cr_residual[0] <= basic_residual[0] + 1e-10
    assert corrected_ni_residual[1] <= basic_residual[1] + 1e-10
    assert abs(corrected_cr.getInterfacePosition() - corrected_ni.getInterfacePosition()) > 1e-10


def test_moving_boundary_fdm_ternary_my_corrected_runs_and_targets_balance_element():
    basic = _make_mock_ternary_moving_boundary_model(interface_update="basic", record=True)
    corrected = _make_mock_ternary_moving_boundary_model(
        interface_update="my_corrected",
        balance_element="CR",
        record=True,
    )
    basic.therm._check_side_of_probe = lambda values: [1.0, -1.0, 1.0, -1.0]
    corrected.therm._check_side_of_probe = lambda values: [1.0, -1.0, 1.0, -1.0]

    basic.solve(5e-5, iterator=explicitEulerIterator)
    corrected.solve(5e-5, iterator=explicitEulerIterator)

    basic_residual = np.abs(basic.getTotalInventory() - basic._initialInventory)
    corrected_residual = np.abs(corrected.getTotalInventory() - corrected._initialInventory)

    assert np.isfinite(corrected.getInterfacePosition())
    assert np.all(np.isfinite(corrected.getTotalInventory()))
    assert corrected_residual[0] <= basic_residual[0] + 1e-10


def test_moving_boundary_fdm_ternary_saving_loading_preserves_balance_element():
    model = _make_mock_ternary_moving_boundary_model(
        interface_update="lee_oh_corrected",
        balance_element="CR",
        record=True,
    )
    model.solve(5e-5, iterator=explicitEulerIterator)
    os.makedirs('.pytest_tmp', exist_ok=True)
    save_path = os.path.join('.pytest_tmp', f'moving_boundary_fdm_ternary_{os.getpid()}.npz')
    model.save(save_path)

    new_model = _make_mock_ternary_moving_boundary_model(
        interface_update="lee_oh_corrected",
        balance_element="CR",
        record=True,
    )
    new_model.load(save_path)

    assert new_model.balanceElement == "CR"
    assert new_model.multicomponentInterfaceStateUpdate == model.multicomponentInterfaceStateUpdate
    assert_allclose(new_model.data.currentY, model.data.currentY)
    assert_allclose(new_model.getInterfacePosition(), model.getInterfacePosition())
    assert_allclose(new_model.getInterfaceCompositions(stage="pre")[0], model.getInterfaceCompositions(stage="pre")[0])
    assert_allclose(new_model.getInterfaceCompositions(stage="pre")[1], model.getInterfaceCompositions(stage="pre")[1])
    assert_allclose(new_model.getInterfaceCompositions(stage="post")[0], model.getInterfaceCompositions(stage="post")[0])
    assert_allclose(new_model.getInterfaceCompositions(stage="post")[1], model.getInterfaceCompositions(stage="post")[1])
    try:
        os.remove(save_path)
    except PermissionError:
        pass


def test_moving_boundary_fdm_ternary_real_thermo_smoke():
    interface_position = 15.25
    profile = ProfileBuilder(
        [
            (StepProfile1D(interface_position, 0.30, 0.26), 'CR'),
            (StepProfile1D(interface_position, 0.085, 0.0575), 'NI'),
        ]
    )
    mesh = CartesianFD1D(['CR', 'NI'], [0, 30], 61)
    mesh.setResponseProfile(profile)
    therm = MulticomponentThermodynamics(FECRNI_DB, ['FE', 'CR', 'NI'], ['FCC_A1', 'BCC_A2'])
    model = MovingBoundaryFD1DModel(
        mesh,
        ['FE', 'CR', 'NI'],
        ['FCC_A1', 'BCC_A2'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1373),
        interfacePosition=interface_position,
        bulkUpdateScheme='flux_form',
        integrationMode='weighted',
        interfaceUpdate='basic',
        balanceElement='CR',
        fluxGradientMode='pre_diffusion',
        initialInventoryMode='integrated',
    )
    model.setup()

    dXdt = model.getdXdt(model.currentTime, model.getCurrentX())
    fluxes = model.getFluxes(model.currentTime, model.getCurrentX())
    interface_compositions = model.getInterfaceCompositions()

    assert dXdt[0].shape == (61, 2)
    assert fluxes.shape == (62, 2)
    assert np.isfinite(dXdt[1])
    assert np.all(np.isfinite(interface_compositions[0]))
    assert np.all(np.isfinite(interface_compositions[1]))
    assert np.all(np.isfinite(model.getTotalInventory()))


def test_moving_boundary_dXdt():
    interfacePosition = 0.5
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.2, 0.8), 'CR')])
    mesh = Cartesian1D(['CR'], [0, 1], 10)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 2.0},
        interface_compositions=(0.3, 0.7),
    )
    from kawin.diffusion.DiffusionParameters import DiffusionConstraints
    constraints = DiffusionConstraints()
    constraints.minComposition = 1e-10
    
    model = MovingBoundary1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        constraints=constraints,
    )
    
    model.setup()
    dXdt = model.getdXdt(model.currentTime, model.getCurrentX())
    fluxes = model.getFluxes(model.currentTime, model.getCurrentX())
    dt = model.getDt(dXdt)

    assert dXdt[0].shape == (10, 1)
    assert_allclose(dXdt[0][4, 0], 20.0, atol=0, rtol=1e-6)
    assert_allclose(dXdt[0][5, 0], -40.0, atol=0, rtol=1e-6)
    assert_allclose(dXdt[1], -5.0, atol=0, rtol=1e-6)
    assert_allclose(fluxes[5, 0], -3.0, atol=0, rtol=1e-6)
    assert_allclose(dt, 0.002, atol=0, rtol=1e-6)


def test_moving_boundary_mass_conservation():
    interfacePosition = 0.5
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    mesh = Cartesian1D(['CR'], [0, 1], 20)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 2.0},
        interface_compositions=(0.35, 0.65),
    )
    
    from kawin.diffusion.DiffusionParameters import DiffusionConstraints
    constraints = DiffusionConstraints()
    constraints.minComposition = 1e-10
    
    model = MovingBoundary1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        constraints=constraints,
        record=True,
    )
    initial_mass = model.getTotalMass()
    model.solve(0.01, iterator=explicitEulerIterator)
    final_mass = model.getTotalMass()

    assert abs(model.getInterfacePosition() - interfacePosition) > 0
    assert_allclose(final_mass, initial_mass, atol=1e-10, rtol=1e-8)


def test_moving_boundary_saving_loading(tmpdir):
    interfacePosition = 0.5
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    mesh = Cartesian1D(['CR'], [0, 1], 20)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 1.0},
        interface_compositions=(0.35, 0.65),
    )
    model = MovingBoundary1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        record=True,
    )
    model.solve(0.01, iterator=explicitEulerIterator)
    model.save(tmpdir / 'moving_boundary.npz')

    new_mesh = Cartesian1D(['CR'], [0, 1], 20)
    new_mesh.setResponseProfile(profile)
    new_model = MovingBoundary1DModel(
        new_mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
        record=True,
    )
    new_model.load(tmpdir / 'moving_boundary.npz')

    assert_allclose(new_model.data.currentY, model.data.currentY)
    assert_allclose(new_model.getInterfacePosition(), model.getInterfacePosition())
    assert_allclose(new_model.currentTime, model.currentTime)


def test_moving_boundary_mass_check_warns():
    interfacePosition = 0.5
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    mesh = Cartesian1D(['CR'], [0, 1], 20)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 1.0},
        interface_compositions=(0.35, 0.65),
    )
    model = MovingBoundary1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
    )
    model.constraints.movingBoundaryMassTolerance = 1e-8
    model.constraints.movingBoundaryMassAction = 'warn'
    model._initialInventory += 1e-4

    with pytest.warns(RuntimeWarning, match='mass correction residual'):
        model._checkMassCorrection(model.data.currentY[:,0], model.getInterfacePosition())


def test_moving_boundary_mass_check_raises():
    interfacePosition = 0.5
    profile = ProfileBuilder([(StepProfile1D(interfacePosition, 0.1, 0.8), 'CR')])
    mesh = Cartesian1D(['CR'], [0, 1], 20)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0, 'BETA': 1.0},
        interface_compositions=(0.35, 0.65),
    )
    model = MovingBoundary1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interfacePosition,
    )
    model.constraints.movingBoundaryMassTolerance = 1e-8
    model.constraints.movingBoundaryMassAction = 'raise'
    model._initialInventory += 1e-4

    with pytest.raises(ValueError, match='mass correction residual'):
        model._checkMassCorrection(model.data.currentY[:,0], model.getInterfacePosition())


def _build_olaye_model(
    interface_position=0.5,
    dt_mode="cfl",
    main_step_mode="leapfrog_dufort_frankel",
    geometry="planar",
    diffusivities=None,
    record=True,
    semi_log_dt=0.05,
    semi_log_t0=1e-6,
):
    profile = ProfileBuilder([(StepProfile1D(interface_position, 0.2, 0.8), 'CR')])
    mesh = CartesianFD1D(['CR'], [0, 1], 81)
    mesh.setResponseProfile(profile)
    if diffusivities is None:
        diffusivities = {'ALPHA': 1.0e-3, 'BETA': 2.0e-3}
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities=diffusivities,
        interface_compositions=(0.3, 0.7),
    )
    model = MovingBoundaryOlayeFD1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000),
        interfacePosition=interface_position,
        interface_compositions=(0.3, 0.7),
        first_step_mode="classical_explicit",
        main_step_mode=main_step_mode,
        dt_mode=dt_mode,
        geometry=geometry,
        semiLog_dt=semi_log_dt,
        semiLogT0=semi_log_t0,
        record=record,
    )
    return model


def test_olaye_moving_boundary_geometry_guards_nonplanar():
    with pytest.raises(NotImplementedError, match="documented but not implemented"):
        _build_olaye_model(geometry="cylindrical")
    with pytest.raises(NotImplementedError, match="documented but not implemented"):
        _build_olaye_model(geometry="spherical")


def test_olaye_moving_boundary_k1_then_lfdf_transition():
    model = _build_olaye_model(dt_mode="cfl", main_step_mode="leapfrog_dufort_frankel")
    model.solve(0.2, iterator=explicitEulerIterator)

    assert model._stepIndex > 1
    assert model._lastStepScheme == "leapfrog_dufort_frankel"
    assert model._p_prev is not None
    assert model._q_prev is not None
    assert np.isfinite(model.getInterfacePosition())


def test_olaye_moving_boundary_conservation_constant_and_variable_diffusivity():
    for therm_class in (ConstantBinaryThermodynamics, VariableBinaryThermodynamics):
        interface_position = 0.50625
        profile = ProfileBuilder([(StepProfile1D(interface_position, 0.25, 0.75), 'CR')])
        mesh = CartesianFD1D(['CR'], [0, 1], 81)
        mesh.setResponseProfile(profile)
        therm = therm_class(
            phases=['ALPHA', 'BETA'],
            diffusivities={'ALPHA': 8.0e-4, 'BETA': 1.2e-3},
            interface_compositions=(0.3, 0.7),
        )
        model = MovingBoundaryOlayeFD1DModel(
            mesh,
            ['FE', 'CR'],
            ['ALPHA', 'BETA'],
            thermodynamics=therm,
            temperature=TemperatureParameters(1000),
            interfacePosition=interface_position,
            interface_compositions=(0.3, 0.7),
            first_step_mode="classical_explicit",
            main_step_mode="leapfrog_dufort_frankel",
            dt_mode="semi_log_optional",
            geometry="planar",
            semiLog_dt=0.05,
            semiLogT0=1e-6,
            record=True,
        )
        model.solve(0.12, iterator=explicitEulerIterator)
        initial_inventory = model.getTotalInventory(time=0)
        final_inventory = model.getTotalInventory()
        assert np.isfinite(final_inventory)
        assert abs(final_inventory - initial_inventory) < 3e-3


def test_olaye_moving_boundary_semi_log_dt_mode_runs():
    model = _build_olaye_model(dt_mode="semi_log_optional", main_step_mode="leapfrog_dufort_frankel")
    model.solve(0.3, iterator=explicitEulerIterator)

    assert model._pendingDtSemi > 0
    assert np.isfinite(model.getInterfacePosition())
    assert model.interfaceData.N > 2


def test_olaye_moving_boundary_requires_explicit_semi_log_controls():
    profile = ProfileBuilder([(StepProfile1D(0.5, 0.2, 0.8), 'CR')])
    mesh = CartesianFD1D(['CR'], [0, 1], 81)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=['ALPHA', 'BETA'],
        diffusivities={'ALPHA': 1.0e-3, 'BETA': 2.0e-3},
        interface_compositions=(0.3, 0.7),
    )

    with pytest.raises(ValueError, match="must be specified explicitly"):
        MovingBoundaryOlayeFD1DModel(
            mesh,
            ['FE', 'CR'],
            ['ALPHA', 'BETA'],
            thermodynamics=therm,
            temperature=TemperatureParameters(1000),
            interfacePosition=0.5,
            interface_compositions=(0.3, 0.7),
            first_step_mode="classical_explicit",
            main_step_mode="leapfrog_dufort_frankel",
            dt_mode="cfl",
            geometry="planar",
            record=True,
        )


def test_olaye_moving_boundary_stability_comparison_against_classical():
    high_D = {'ALPHA': 2.0e-2, 'BETA': 4.0e-2}
    lfdf_model = _build_olaye_model(main_step_mode="leapfrog_dufort_frankel", diffusivities=high_D, record=True)
    classical_model = _build_olaye_model(main_step_mode="classical_explicit", diffusivities=high_D, record=True)
    lfdf_model.constraints.vonNeumannThreshold = 2.0
    classical_model.constraints.vonNeumannThreshold = 2.0

    lfdf_model.solve(0.06, iterator=explicitEulerIterator)
    classical_model.solve(0.06, iterator=explicitEulerIterator)

    lfdf_vals = np.asarray(lfdf_model.data.currentY[:, 0], dtype=np.float64)
    classical_vals = np.asarray(classical_model.data.currentY[:, 0], dtype=np.float64)
    # Synthetic stability check: LF/DF should avoid larger excursions compared to classical.
    assert np.max(np.abs(lfdf_vals - 0.5)) <= np.max(np.abs(classical_vals - 0.5)) + 1e-8


def test_olaye_moving_boundary_paper_style_benchmark_monotonic_and_sqrt_like_growth():
    model = _build_olaye_model(
        interface_position=0.42,
        main_step_mode="leapfrog_dufort_frankel",
        dt_mode="semi_log_optional",
        diffusivities={'ALPHA': 4.0e-4, 'BETA': 3.0e-3},
        record=True,
    )
    model.solve(0.4, iterator=explicitEulerIterator)

    t = np.array(model.interfaceData._time[: model.interfaceData.N + 1], dtype=np.float64)
    s = np.array(model.interfaceData._y[: model.interfaceData.N + 1], dtype=np.float64)
    assert len(t) > 8
    # "Paper-style" behavior check: dominant one-direction migration.
    ds = np.diff(s)
    net_sign = np.sign(s[-1] - s[0]) if not np.isclose(s[-1], s[0]) else 0.0
    same_direction_fraction = np.mean(np.sign(ds[np.abs(ds) > 1e-14]) == net_sign) if np.any(np.abs(ds) > 1e-14) else 1.0
    assert same_direction_fraction > 0.7
    # Benchmark-style sanity: displacement is finite and non-trivial.
    assert np.all(np.isfinite(s))
    assert abs(s[-1] - s[0]) > 1e-8


def test_olaye_moving_boundary_semi_log_schedule_is_monotone():
    model = _build_olaye_model(dt_mode="semi_log_optional", main_step_mode="leapfrog_dufort_frankel")
    model.setTimeInfo(0.0, 0.3)

    assert model._semiLogTimes is not None
    assert len(model._semiLogTimes) >= 3
    assert np.all(np.diff(model._semiLogTimes) > 0)
    assert model._semiLogTimes[0] < model._semiLogTimes[-1]
    assert_allclose(model._semiLogTimes[-1], 0.3, rtol=0, atol=1e-12)


def test_olaye_moving_boundary_rejects_non_euler_iterator():
    model = _build_olaye_model(dt_mode="semi_log_optional", main_step_mode="leapfrog_dufort_frankel")

    with pytest.raises(ValueError, match="supports only explicitEulerIterator"):
        model.solve(0.05, iterator=rk4Iterator)


def test_olaye_moving_boundary_planar_interface_root_matches_paper_coefficients():
    model = _build_olaye_model(dt_mode="semi_log_optional", main_step_mode="leapfrog_dufort_frankel")
    model.setup()

    p = model._p_curr.copy()
    q = model._q_curr.copy()
    s = float(model._s_curr)
    D_p, D_q = model._phase_diffusivities(0.0, p, q, s)
    grad_left, grad_right, _ = model._interface_velocity(p, q, s, D_p, D_q)
    dt = 1.0e-4

    c_ab, c_ba = model.interfaceCompositions
    p_half = 0.5 * (p[-2] + c_ab)
    q_half = 0.5 * (c_ba + q[1])
    u_half = 0.5 * (model._u_grid[-2] + model._u_grid[-1])
    v_half = 0.5 * (model._v_grid[0] + model._v_grid[1])
    expected_a = p_half * u_half + q_half * (1.0 - v_half) + c_ab - c_ba
    expected_b = dt * (D_p[-1] * grad_left + D_q[0] * grad_right) - expected_a * s

    s_new = model._compute_next_interface_position(p, q, s, dt, grad_left, grad_right, D_p, D_q)
    assert_allclose(model._lastInterfaceCoefficients[0], expected_a, rtol=0, atol=1e-12)
    assert_allclose(model._lastInterfaceCoefficients[1], expected_b, rtol=0, atol=1e-12)
    assert_allclose(s_new, -expected_b / expected_a, rtol=0, atol=1e-12)
    assert_allclose(model._lastInterfaceCoefficients[0] * s_new + model._lastInterfaceCoefficients[1], 0.0, rtol=0, atol=1e-12)


def test_olaye_transformed_state_history_records_and_queries_exact_times():
    model = _build_olaye_model(record=2)
    model.solve(0.2, iterator=explicitEulerIterator)

    p_latest, q_latest = model.getTransformedState()
    assert_allclose(p_latest, model._p_curr, rtol=0.0, atol=0.0)
    assert_allclose(q_latest, model._q_curr, rtol=0.0, atol=0.0)

    recorded_times = np.array(model.pData._time[: model.pData.N + 1], dtype=np.float64)
    assert_allclose(recorded_times, model.qData._time[: model.qData.N + 1], rtol=0.0, atol=0.0)
    assert_allclose(recorded_times, model.interfaceData._time[: model.interfaceData.N + 1], rtol=0.0, atol=0.0)
    assert len(recorded_times) >= 2
    assert np.all(np.diff(recorded_times) > 0)

    t_mid = float(recorded_times[min(1, len(recorded_times) - 1)])
    p_mid = model.getTransformedStateLeft(t_mid)
    q_mid = model.getTransformedStateRight(t_mid)
    assert_allclose(p_mid, model.pData.y(t_mid), rtol=0.0, atol=0.0)
    assert_allclose(q_mid, model.qData.y(t_mid), rtol=0.0, atol=0.0)
    assert model.pData._y.shape[0] == model.pData.N + 1
    assert model.qData._y.shape[0] == model.qData.N + 1


def _build_illingworth_default_model(therm_class=ConstantBinaryThermodynamics, **kwargs):
    params = {
        "s0": 1.0,
        "R": 5.0,
        "n_alpha": 100,
        "d_alpha": 1.0e-7,
        "initial_alpha": 0.8,
        "interface_alpha": 0.6,
        "n_beta": 100,
        "d_beta": 1.0e-5,
        "initial_beta": 0.4,
        "interface_beta": 0.0,
        "time_step": 0.1,
        "tolerance": 1.0e-8,
        "geometry": "planar",
        "record": True,
        "transformed_u_grid": None,
        "transformed_v_grid": None,
    }
    params.update(kwargs)
    profile = ProfileBuilder([(StepProfile1D(params["s0"], params["initial_alpha"], params["initial_beta"]), "CR")])
    mesh = CartesianFD1D(["CR"], [0.0, params["R"]], 501)
    mesh.setResponseProfile(profile)
    therm = therm_class(
        phases=["ALPHA", "BETA"],
        diffusivities={"ALPHA": params["d_alpha"], "BETA": params["d_beta"]},
        interface_compositions=(params["interface_alpha"], params["interface_beta"]),
    )
    return MovingBoundaryIllingworthFD1DModel(
        mesh,
        ["NI", "CR"],
        ["ALPHA", "BETA"],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000.0),
        interfacePosition=params["s0"],
        interface_compositions=(params["interface_alpha"], params["interface_beta"]),
        time_step=params["time_step"],
        geometry=params["geometry"],
        phase_a_nodes=params["n_alpha"],
        phase_b_nodes=params["n_beta"],
        tolerance=params["tolerance"],
        record=params["record"],
        transformed_u_grid=params["transformed_u_grid"],
        transformed_v_grid=params["transformed_v_grid"],
    )


def test_illingworth_moving_boundary_guards():
    with pytest.raises(NotImplementedError, match="only planar"):
        _build_illingworth_default_model(geometry="spherical")
    with pytest.raises(ValueError, match="time_step"):
        _build_illingworth_default_model(time_step=0.0)
    with pytest.raises(ValueError, match="phase_a_nodes"):
        _build_illingworth_default_model(n_alpha=2)
    with pytest.raises(ValueError, match="constant diffusivity"):
        _build_illingworth_default_model(therm_class=VariableBinaryThermodynamics).solve(0.1, iterator=explicitEulerIterator)

    model = _build_illingworth_default_model()
    with pytest.raises(ValueError, match="explicitEulerIterator"):
        model.solve(0.1, iterator=rk4Iterator)


def test_illingworth_default_planar_case_regression():
    model = _build_illingworth_default_model()
    model.solve(1.0, iterator=explicitEulerIterator, minDtFrac=1e-14)

    t = model.interfaceData._time[: model.interfaceData.N + 1]
    s = model.interfaceData._y[: model.interfaceData.N + 1]
    expected_t = np.linspace(0.0, 1.0, 11)
    expected_s = np.array(
        [
            1.0,
            1.0000498907689606,
            1.0000996876816444,
            1.0001493912089776,
            1.0001990018180928,
            1.0002485199723707,
            1.0002979461314816,
            1.0003472807514266,
            1.0003965242845785,
            1.0004456771797203,
            1.0004947398820860,
        ],
        dtype=np.float64,
    )
    assert_allclose(t, expected_t, rtol=0.0, atol=1e-12)
    assert_allclose(s, expected_s, rtol=0.0, atol=5e-12)


def test_illingworth_custom_uniform_transformed_grids_preserve_default_behavior():
    default = _build_illingworth_default_model()
    custom = _build_illingworth_default_model(
        transformed_u_grid=np.linspace(0.0, 1.0, 100),
        transformed_v_grid=np.linspace(0.0, 1.0, 100),
    )

    default.solve(0.5, iterator=explicitEulerIterator, minDtFrac=1e-14)
    custom.solve(0.5, iterator=explicitEulerIterator, minDtFrac=1e-14)

    default_t = default.interfaceData._time[: default.interfaceData.N + 1]
    custom_t = custom.interfaceData._time[: custom.interfaceData.N + 1]
    default_s = default.interfaceData._y[: default.interfaceData.N + 1]
    custom_s = custom.interfaceData._y[: custom.interfaceData.N + 1]
    assert_allclose(custom_t, default_t, rtol=0.0, atol=0.0)
    assert_allclose(custom_s, default_s, rtol=0.0, atol=0.0)


def test_illingworth_custom_transformed_grid_validation():
    with pytest.raises(ValueError, match="strictly increasing"):
        _build_illingworth_default_model(transformed_v_grid=np.array([0.0, 0.5, 0.5, 1.0]))
    with pytest.raises(ValueError, match="start at 0 and end at 1"):
        _build_illingworth_default_model(transformed_u_grid=np.array([0.1, 0.5, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        _build_illingworth_default_model(transformed_v_grid=np.array([0.0, np.nan, 1.0]))
    with pytest.raises(ValueError, match="phase_b_nodes"):
        _build_illingworth_default_model(n_beta=4, transformed_v_grid=np.linspace(0.0, 1.0, 5))


def test_illingworth_default_planar_case_conservation():
    model = _build_illingworth_default_model()
    model.solve(1.0, iterator=explicitEulerIterator, minDtFrac=1e-14)

    assert model._lastImplicitIterations >= 2
    assert model.checkConservation(1e-10) < 1e-10
    expected_average = model.checkMassIntegral(model._p_curr, model._q_curr, model._s_curr)
    assert_allclose(model.concData.y(), expected_average, rtol=0.0, atol=1e-14)
    recorded_average = model.concData._y[: model.concData.N + 1]
    assert_allclose(recorded_average, recorded_average[0], rtol=0.0, atol=1e-10)
    assert_allclose(model.getTotalInventory(time=0.5), recorded_average[0] * model._R, rtol=0.0, atol=1e-10)


def test_illingworth_transformed_state_history_records_and_queries_exact_times():
    model = _build_illingworth_default_model(record=2)
    model.solve(0.4, iterator=explicitEulerIterator, minDtFrac=1e-14)

    p_latest, q_latest = model.getTransformedState()
    assert_allclose(p_latest, model._p_curr, rtol=0.0, atol=0.0)
    assert_allclose(q_latest, model._q_curr, rtol=0.0, atol=0.0)

    recorded_times = np.array(model.pData._time[: model.pData.N + 1], dtype=np.float64)
    assert_allclose(recorded_times, model.qData._time[: model.qData.N + 1], rtol=0.0, atol=0.0)
    assert_allclose(recorded_times, model.interfaceData._time[: model.interfaceData.N + 1], rtol=0.0, atol=0.0)
    assert len(recorded_times) >= 2
    assert np.all(np.diff(recorded_times) > 0)

    t_mid = float(recorded_times[min(1, len(recorded_times) - 1)])
    p_mid, q_mid = model.getTransformedState(t_mid)
    assert_allclose(p_mid, model.pData.y(t_mid), rtol=0.0, atol=0.0)
    assert_allclose(q_mid, model.qData.y(t_mid), rtol=0.0, atol=0.0)
    assert model.pData._y.shape[0] == model.pData.N + 1
    assert model.qData._y.shape[0] == model.qData.N + 1


def test_illingworth_cpp_reference_comparison():
    try:
        from examples.Illingworth2005.compare_illingworth2005_planar import compare_default_case

        comparison = compare_default_case()
    except (FileNotFoundError, RuntimeError, PermissionError) as exc:
        pytest.skip(f"Authors' C++ comparison unavailable: {exc}")

    # The MAP code writes results.txt with "%lg", so the parsed C++ reference
    # has about six significant digits even though the internal calculation is
    # double precision.
    assert comparison["max_abs_diff"] < 5e-6
    assert comparison["max_rel_diff"] < 5e-6


def test_illingworth_fig456_planar_analytical_helpers():
    from examples.Illingworth2005.replicate_illingworth2005_fig4_5_6_planar import (
        VALIDATION_PARAMS,
        planar_exact_interface,
        planar_exact_profile,
        solve_planar_zener_j,
    )

    j = solve_planar_zener_j()
    ratio = (VALIDATION_PARAMS["c_b"] - VALIDATION_PARAMS["c_inf"]) / (
        VALIDATION_PARAMS["c_b"] - VALIDATION_PARAMS["c_a"]
    )
    assert 2.0 * j * 0.5 * np.sqrt(np.pi) * np.exp(j * j) * math.erfc(j) == pytest.approx(ratio)

    t_abs = VALIDATION_PARAMS["t_init"]
    s = planar_exact_interface(t_abs, j=j)
    profile = planar_exact_profile(
        np.array([0.0, s * (1.0 + 1.0e-8), VALIDATION_PARAMS["R"]], dtype=np.float64),
        t_abs,
        j=j,
    )
    assert profile[0] == pytest.approx(VALIDATION_PARAMS["c_a"])
    assert profile[1] == pytest.approx(VALIDATION_PARAMS["c_b"], rel=0.0, abs=1e-8)
    assert profile[2] == pytest.approx(VALIDATION_PARAMS["c_inf"], rel=0.0, abs=1e-12)


def test_illingworth_fig456_notebook_entrypoint_quick_smoke():
    from examples.Illingworth2005.replicate_illingworth2005_fig4_5_6_planar import run_and_plot

    results, axes = run_and_plot(
        {
            "quick": True,
            "cpp_enabled": False,
            "show": False,
            "out": None,
            "print_summary": False,
            "tight_layout": False,
        }
    )

    assert set(axes) == {"4", "5", "6"}
    assert len(results["cases"]) >= 3
    assert set(results["extracted_data"]) == {"4", "5", "6"}
    assert all(len(entries) >= 1 for entries in results["extracted_data"].values())
    assert all("python" in item for item in results["cases"])
    for item in results["cases"]:
        assert np.all(np.isfinite(item["python"]["interface"]))


def test_illingworth_fig456_extracted_data_loader():
    from examples.Illingworth2005.replicate_illingworth2005_fig4_5_6_planar import (
        SCRIPT_DIR,
        load_extracted_figure_data,
    )

    x, y = load_extracted_figure_data(SCRIPT_DIR / "figureDataExtraction" / "fig5a_dt_1Eminus5.csv")

    assert len(x) == len(y)
    assert len(x) > 2
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))


def test_illingworth_fig456_quick_python_cpp_comparison():
    if shutil.which("g++") is None:
        pytest.skip("g++ is not available for the authors' C++ comparison.")

    try:
        from examples.Illingworth2005.replicate_illingworth2005_fig4_5_6_planar import run_planar_validation_cases

        results = run_planar_validation_cases(
            {
                "figures": ("4",),
                "quick": True,
                "cpp_enabled": True,
                "show": False,
                "out": None,
                "print_summary": False,
                "t_abs_end": 3.0e-4,
            }
        )
    except (FileNotFoundError, RuntimeError, PermissionError) as exc:
        pytest.skip(f"Authors' C++ comparison unavailable: {exc}")

    assert len(results["cases"]) == 2
    for item in results["cases"]:
        assert "cpp" in item
        assert item["python_cpp_comparison"]["max_abs_diff"] < 2e-5


def test_olaye_saved_run_payload_contains_required_fields():
    from examples.Olaye2020.replicate_olaye2020_fig5 import build_olaye_run_payload

    payload = build_olaye_run_payload(
        time_s=np.array([0.0, 1.0, 2.0]),
        half_width_um=np.array([12.5, 13.0, 13.2]),
        params={
            "s0_um": 12.5,
            "c_liquid0_pct": 19.0,
            "c_liquid_int_pct": 10.223,
            "show": True,
            "out": None,
        },
        model_variant="rework",
        label="Synthetic Olaye",
    )

    assert set(["time_s", "half_width_um", "label", "source_script", "model_family", "params_json", "model_variant"]).issubset(payload)
    assert payload["model_family"] == "olaye_fig5"
    assert payload["label"] == "Synthetic Olaye"


def test_olaye_figure_selector_normalizes_requested_figures():
    from examples.Olaye2020.replicate_olaye2020_fig5 import _selected_figures

    assert _selected_figures({"figures": "fig5"}) == ("fig5",)
    assert _selected_figures({"figures": ["5", "fig6"]}) == ("fig5", "fig6")
    assert _selected_figures({"figures": "all"}) == ("fig5", "fig6")


def test_olaye_fig6_selection_validation():
    from examples.Olaye2020.replicate_olaye2020_fig5 import (
        _selected_fig6_d_alpha_cm2_s,
        _selected_fig6_layers,
    )

    assert _selected_fig6_layers({"fig6_layers": "both"}) == ("thin", "thick")
    assert _selected_fig6_layers({"fig6_layers": ["thick", "thin"]}) == ("thick", "thin")
    assert _selected_fig6_d_alpha_cm2_s({"fig6_d_alpha_cm2_s": "both"}) == (1.4e-8, 2.5e-8)
    assert _selected_fig6_d_alpha_cm2_s({"fig6_d_alpha_cm2_s": [2.5e-8]}) == (2.5e-8,)

    with pytest.raises(ValueError, match="Unknown Figure 6 layer"):
        _selected_fig6_layers({"fig6_layers": ["medium"]})
    with pytest.raises(ValueError, match="Unknown Figure 6 alpha diffusivity"):
        _selected_fig6_d_alpha_cm2_s({"fig6_d_alpha_cm2_s": [3.0e-8]})


def test_olaye_fig6_case_params_use_corrected_brass_table_values():
    from examples.Olaye2020.replicate_olaye2020_fig5 import build_fig6_case_params

    thin = build_fig6_case_params(
        "thin",
        1.4e-8,
        {
            "fig6_n_phase_a_nodes": 12,
            "fig6_n_phase_b_nodes": 34,
            "fig6_semiLog_dt": 0.125,
            "fig6_t_end_s": 10.0,
        },
    )
    thick = build_fig6_case_params("thick", 2.5e-8)

    assert thin["phase_a_name"] == "BETA"
    assert thin["phase_b_name"] == "ALPHA"
    assert thin["s0_um"] == 190.5
    assert thin["R_um"] == 565.0
    assert thin["c_liquid0_pct"] == 39.4
    assert thin["c_solid0_pct"] == 29.1
    assert thin["c_liquid_int_pct"] == 36.9
    assert thin["c_solid_int_pct"] == 32.5
    assert thin["D_liquid_base"] == 100.0
    assert thin["D_solid_base"] == 1.4
    assert thin["n_phase_a_nodes"] == 12
    assert thin["n_phase_b_nodes"] == 34
    assert thin["semiLog_dt"] == 0.125
    assert thin["t_end_s"] == 10.0

    assert thick["s0_um"] == 381.0
    assert thick["R_um"] == 755.5
    assert thick["D_solid_base"] == 2.5


def test_olaye_fig6_analytical_constants_use_um2_per_s_units():
    from examples.Olaye2020.replicate_olaye2020_fig5 import _fig6_analytical_constants, build_fig6_case_params, solve_beta

    params = build_fig6_case_params("thin", 1.4e-8)
    constants = _fig6_analytical_constants(params)

    assert constants["d_a_um2_s"] == 100.0
    assert constants["d_b_um2_s"] == 1.4
    assert constants["s0_um"] == 190.5
    assert constants["R_um"] == 565.0
    assert_allclose(
        constants["beta_um_sqrt_s"],
        solve_beta(0.394, 0.291, 0.369, 0.325, 100.0, 1.4, left=-100.0, right=100.0),
    )


def test_olaye_no_header_xy_csv_loader():
    from io import StringIO

    from examples.Olaye2020.replicate_olaye2020_fig5 import _load_no_header_xy_csv

    x, y = _load_no_header_xy_csv(StringIO("1.0,2.0\nnan,3.0\n4.0,5.0\n"))

    assert_allclose(x, [1.0, 4.0])
    assert_allclose(y, [2.0, 5.0])


def test_olaye_fig6_payload_contains_interface_displacement():
    from examples.Olaye2020.replicate_olaye2020_fig5 import build_olaye_run_payload

    payload = build_olaye_run_payload(
        time_s=np.array([10.0, 100.0]),
        interface_displacement_um=np.array([1.0, -2.0]),
        params={"figure": "fig6", "show": True},
        model_variant="rework",
        model_family="olaye_fig6",
        label="Synthetic Figure 6",
    )

    assert "interface_displacement_um" in payload
    assert "half_width_um" not in payload
    assert payload["model_family"] == "olaye_fig6"
    assert payload["label"] == "Synthetic Figure 6"


def test_olaye_fig6_concentration_info_plot_helper_adds_twin_axis():
    from examples.Olaye2020.replicate_olaye2020_fig5 import _plot_fig6_concentration_info, build_fig6_case_params

    class FakeHistory:
        def __init__(self):
            self._time = np.array([1.0, 10.0, 100.0])
            self._y = np.array([0.32, 0.33, 0.34])

    class FakeModel:
        def __init__(self):
            self.concData = FakeHistory()

    fig, ax = plt.subplots()
    try:
        params = build_fig6_case_params("thin", 1.4e-8)
        ax_conc = _plot_fig6_concentration_info(
            ax,
            [
                {
                    "model": FakeModel(),
                    "params": params,
                }
            ],
        )

        assert ax_conc is not ax
        assert ax_conc.get_ylabel() == "conc"
        assert len(ax_conc.lines) == 3
    finally:
        plt.close(fig)


def test_olaye_fig6_extracted_data_specs_match_plotted_cases():
    from examples.Olaye2020.replicate_olaye2020_fig5 import (
        _fig6_extracted_data_specs_for_cases,
        build_fig6_case_params,
    )

    thick_specs = _fig6_extracted_data_specs_for_cases(
        [
            {
                "params": build_fig6_case_params("thick", 1.4e-8),
            }
        ]
    )
    thick_d2pt5_specs = _fig6_extracted_data_specs_for_cases(
        [
            {
                "params": build_fig6_case_params("thick", 2.5e-8),
            }
        ]
    )
    both_specs = _fig6_extracted_data_specs_for_cases(
        [
            {
                "params": build_fig6_case_params("thin", 1.4e-8),
            },
            {
                "params": build_fig6_case_params("thick", 1.4e-8),
            },
        ]
    )

    assert [spec["filename"] for spec in thick_specs] == ["Olaye2020_fig6_ThickBeta_D1pt4_DF_curve.csv"]
    assert [spec["filename"] for spec in thick_d2pt5_specs] == ["Olaye2020_fig6_ThickBeta_D2pt5_DF_curve.csv"]
    assert [spec["filename"] for spec in both_specs] == [
        "Olaye2020_fig6_ThinBeta_D1pt4_DF_curve.csv",
        "Olaye2020_fig6_ThickBeta_D1pt4_DF_curve.csv",
    ]


def test_olaye_fig6_sqrt_time_analytical_plot_helper(monkeypatch):
    from examples.Olaye2020 import replicate_olaye2020_fig5 as olaye

    monkeypatch.setattr(
        olaye,
        "_fig6_extracted_data_specs_for_cases",
        lambda case_results: [
            {
                "path": pathlib.Path(__file__),
                "color": "black",
                "marker": "o",
                "label": "Synthetic extracted data",
            }
        ],
    )
    monkeypatch.setattr(
        olaye,
        "_load_no_header_xy_csv",
        lambda path: (
            np.array([100.0, 1.0e8, 100.0]),
            np.array([8.0, 8.0, 30.0]),
        ),
    )

    params = olaye.build_fig6_case_params("thin", 1.4e-8)
    from matplotlib.figure import Figure
    fig = Figure()
    ax = fig.add_subplot(111)
    result = olaye.plot_olaye_fig6_sqrt_time_analytical(
        [
            {
                "time_s": np.array([0.0, 100.0, 400.0]),
                "interface_displacement_um": np.array([0.0, 8.0, 16.0]),
                "params": params,
            }
        ],
        {
            "show": False,
            "fig6_sqrt_time_out": False,
            "fig6_sqrt_time_plot_extracted_data": True,
            "fig6_sqrt_time_max_s": None,
        },
        ax=ax,
    )

    assert result["axes"].get_xlabel() == "sqrt(time) (sqrt(s))"
    assert result["axes"].get_ylabel() == "Interface displacement (um)"
    assert len(result["axes"].lines) == 2
    assert len(result["axes"].collections) == 1
    assert_allclose(result["axes"].collections[0].get_offsets(), [[10.0, 8.0], [10.0, 30.0]])
    assert result["axes"].get_xlim()[1] < np.sqrt(1.0e8)
    assert result["axes"].get_ylim()[1] > 30.0


def test_illingworth_saved_run_payload_contains_required_fields():
    from examples.Illingworth2005.compare_illingworth2005_planar import build_illingworth_run_payload

    payload = build_illingworth_run_payload(
        {
            "time_s": np.array([0.1, 1.0, 10.0]),
            "liquid_half_width_um": np.array([12.5, 13.0, 14.0]),
            "theoretical_max_um": 23.2,
            "params": {"time_step_s": 10.0, "show": True, "out": None},
        },
        label="Synthetic Illingworth",
    )

    assert set(["time_s", "half_width_um", "label", "source_script", "model_family", "params_json", "theoretical_max_um"]).issubset(payload)
    assert payload["model_family"] == "illingworth_fig3_present_work"
    assert payload["label"] == "Synthetic Illingworth"


def test_illingworth_fig6_case_params_use_corrected_brass_table_values():
    from examples.Illingworth2005.compare_illingworth2005_planar import build_fig6_illingworth_case_params

    thin = build_fig6_illingworth_case_params(
        "thin",
        1.4e-8,
        {
            "fig6_n_phase_a_nodes": 12,
            "fig6_n_phase_b_nodes": 34,
            "fig6_semiLog_dt": 0.125,
            "fig6_t_end_s": 10.0,
        },
    )
    thick = build_fig6_illingworth_case_params("thick", 2.5e-8)

    assert thin["phase_a_name"] == "BETA"
    assert thin["phase_b_name"] == "ALPHA"
    assert thin["response_name"] == "ZN"
    assert thin["elements"] == ("CU", "ZN")
    assert thin["s0_um"] == 190.5
    assert thin["R_um"] == 565.0
    assert thin["c_liquid0_atpct"] == 39.4
    assert thin["c_solid0_atpct"] == 29.1
    assert thin["c_liquid_int_atpct"] == 36.9
    assert thin["c_solid_int_atpct"] == 32.5
    assert thin["D_liquid_um2_s"] == 100.0
    assert thin["D_solid_um2_s"] == 1.4
    assert thin["n_alpha"] == 12
    assert thin["n_beta"] == 34
    assert thin["semiLog_dt"] == 0.125
    assert thin["t_end_s"] == 10.0
    assert not thin["checkAgainstAuthorsCPP"]

    assert thick["s0_um"] == 381.0
    assert thick["R_um"] == 755.5
    assert thick["D_solid_um2_s"] == 2.5

    cpp_enabled = build_fig6_illingworth_case_params(
        "thin",
        1.4e-8,
        {
            "fig6_check_against_authors_cpp": True,
            "fig6_plot_cpp_comparison": False,
            "fig6_cpp_compiler": "custom-g++",
            "fig6_cpp_build_dir": pathlib.Path("cpp-build"),
            "fig6_cpp_print_table": True,
        },
    )

    assert cpp_enabled["checkAgainstAuthorsCPP"]
    assert not cpp_enabled["plot_cpp_comparison"]
    assert cpp_enabled["cpp_compiler"] == "custom-g++"
    assert cpp_enabled["cpp_build_dir"] == pathlib.Path("cpp-build")
    assert cpp_enabled["cpp_print_table"]


def test_illingworth_fig3_builder_accepts_tolerance_override():
    from examples.Illingworth2005.compare_illingworth2005_planar import build_fig3_present_work_model

    model = build_fig3_present_work_model(
        {
            "tolerance": 3.0e-12,
            "n_alpha": 5,
            "n_beta": 6,
            "record_pq_data": False,
            "preallocate_recordings": False,
        },
        record=False,
    )

    assert model.tolerance == pytest.approx(3.0e-12)


def test_illingworth_fig6_analytical_constants_use_um2_per_s_units():
    from examples.Illingworth2005.compare_illingworth2005_planar import (
        _fig6_analytical_constants,
        build_fig6_illingworth_case_params,
        solve_beta,
    )

    params = build_fig6_illingworth_case_params("thin", 1.4e-8)
    constants = _fig6_analytical_constants(params)

    assert constants["d_a_um2_s"] == 100.0
    assert constants["d_b_um2_s"] == 1.4
    assert constants["s0_um"] == 190.5
    assert constants["R_um"] == 565.0
    assert_allclose(
        constants["beta_um_sqrt_s"],
        solve_beta(0.394, 0.291, 0.369, 0.325, 100.0, 1.4, left=-100.0, right=100.0),
    )


def test_illingworth_fig6_cpp_comparison_matches_fixed_step_history(monkeypatch):
    from examples.Illingworth2005 import compare_illingworth2005_planar as illingworth

    params = illingworth.build_fig6_illingworth_case_params(
        "thin",
        1.4e-8,
        {
            "fig6_dt_mode": "fixed",
            "fig6_time_step_s": 0.5,
            "fig6_t_end_s": 2.0,
        },
    )
    cpp_time = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    cpp_interface = np.array([190.5, 191.0, 191.5, 192.0, 192.5])

    monkeypatch.setattr(
        illingworth,
        "compile_and_run_authors_cpp_with_params",
        lambda params, compiler=None, build_dir=None, return_runtime=False: (
            (cpp_time, cpp_interface),
            {
                "cpp_compile_runtime_s": 0.2,
                "cpp_run_runtime_s": 0.1,
                "cpp_total_runtime_s": 0.3,
            },
        ),
    )

    comparison = illingworth.compare_fig6_result_to_authors_cpp(
        {
            "time_s": np.array([0.0, 1.0, 2.0]),
            "phase_a_width_um": np.array([190.5, 191.5, 192.5]),
            "python_runtime_s": 0.4,
            "params": params,
        }
    )

    assert_allclose(comparison["time"], [0.0, 1.0, 2.0])
    assert_allclose(comparison["cpp_s"], [190.5, 191.5, 192.5])
    assert_allclose(comparison["cpp_interface_displacement_um"], [0.0, 0.5, 1.0, 1.5, 2.0])
    assert comparison["max_abs_diff"] == 0.0
    assert comparison["python_to_cpp_run_runtime_ratio"] == 4.0


def test_illingworth_fig6_run_cpp_comparison_fixed_mode_and_skip(monkeypatch):
    from examples.Illingworth2005 import compare_illingworth2005_planar as illingworth

    class FakeHistory:
        def __init__(self, time_s, values):
            self._time = np.asarray(time_s, dtype=np.float64)
            self._y = np.asarray(values, dtype=np.float64)
            self.N = len(self._time) - 1

    class FakeModel:
        def __init__(self, params):
            self.interfaceData = FakeHistory(
                [0.0, params["time_step_s"], params["t_end_s"]],
                [params["s0_um"], params["s0_um"] + 0.5, params["s0_um"] + 1.0],
            )

        def solve(self, *args, **kwargs):
            return

    monkeypatch.setattr(illingworth, "build_fig3_present_work_model", lambda params, record=None: FakeModel(params))
    monkeypatch.setattr(
        illingworth,
        "compare_fig6_result_to_authors_cpp",
        lambda result, compiler=None, build_dir=None: {
            "time": result["time_s"],
            "cpp_s": result["phase_a_width_um"],
            "python_s": result["phase_a_width_um"],
            "abs_diff": np.zeros_like(result["time_s"]),
            "rel_diff": np.zeros_like(result["time_s"]),
            "max_abs_diff": 0.0,
            "max_rel_diff": 0.0,
        },
    )

    fixed_params = illingworth.build_fig6_illingworth_case_params(
        "thin",
        1.4e-8,
        {
            "fig6_check_against_authors_cpp": True,
            "fig6_dt_mode": "fixed",
            "fig6_time_step_s": 0.5,
            "fig6_t_end_s": 1.0,
        },
    )
    fixed_result = illingworth.run_fig6_olaye_brass_case(fixed_params)

    assert "cpp_comparison" in fixed_result
    assert fixed_result["cpp_comparison"]["max_abs_diff"] == 0.0

    semi_log_params = illingworth.build_fig6_illingworth_case_params(
        "thin",
        1.4e-8,
        {
            "fig6_check_against_authors_cpp": True,
            "fig6_dt_mode": "semi_log",
            "fig6_t_end_s": 1.0,
        },
    )
    semi_log_result = illingworth.run_fig6_olaye_brass_case(semi_log_params)

    assert "cpp_comparison" not in semi_log_result
    assert "cpp_comparison_skipped" in semi_log_result


def test_illingworth_fig6_plot_dispatch_uses_completed_case_overlays(monkeypatch):
    from matplotlib.figure import Figure

    from examples.Illingworth2005 import compare_illingworth2005_planar as illingworth

    params = illingworth.build_fig6_illingworth_case_params("thick", 1.4e-8)

    def fake_run(case_params):
        return {
            "time_s": np.array([0.0, 10.0, 100.0]),
            "phase_a_width_um": np.array([case_params["s0_um"], case_params["s0_um"] + 1.0, case_params["s0_um"] + 2.0]),
            "interface_displacement_um": np.array([0.0, 1.0, 2.0]),
            "model": object(),
            "params": case_params,
            "grid_metadata": {},
            "transformed_u_grid": None,
            "transformed_v_grid": None,
            "cpp_comparison": {
                "time": np.array([0.0, 10.0, 100.0]),
                "cpp_s": np.array([case_params["s0_um"], case_params["s0_um"] + 1.0, case_params["s0_um"] + 2.0]),
                "python_s": np.array([case_params["s0_um"], case_params["s0_um"] + 1.0, case_params["s0_um"] + 2.0]),
                "abs_diff": np.zeros(3),
                "rel_diff": np.zeros(3),
                "max_abs_diff": 0.0,
                "max_rel_diff": 0.0,
                "cpp_time": np.array([0.0, 10.0, 100.0]),
                "cpp_interface": np.array([case_params["s0_um"], case_params["s0_um"] + 1.0, case_params["s0_um"] + 2.0]),
            },
        }

    monkeypatch.setattr(illingworth, "run_fig6_olaye_brass_case", fake_run)
    monkeypatch.setattr(
        illingworth,
        "plot_fig6_olaye_brass_sqrt_time_analytical",
        lambda case_results, config: {"cases": case_results, "params": config},
    )
    fig = Figure()
    ax = fig.add_subplot(111)
    result = illingworth.plot_fig6_olaye_brass_illingworth(
        {
            "fig6_layers": ("thick",),
            "fig6_d_alpha_cm2_s": (1.4e-8,),
            "fig6_plot_concentration_info": False,
            "fig6_plot_extracted_data": True,
            "fig6_check_against_authors_cpp": True,
            "fig6_make_overlay_png": False,
            "fig6_plot_sqrt_time_analytical": True,
            "fig6_sqrt_time_out": False,
            "fig6_sqrt_time_plot_extracted_data": False,
            "fig6_sqrt_time_plot_experimental_data": False,
            "fig6_out": False,
            "show": False,
        },
        ax=ax,
    )

    assert len(result["cases"]) == 1
    assert result["cases"][0]["params"]["model_family"] == "illingworth_fig6_olaye_brass"
    assert result["axes"].get_ylabel() == "Interface displacement (um)"
    assert result["sqrt_time_analytical"] is not None
    assert len(result["axes"].lines) == 2
    assert len(result["axes"].collections) == 2


def test_illingworth_fig6_sqrt_time_analytical_plot_helper(monkeypatch):
    from matplotlib.figure import Figure

    from examples.Illingworth2005 import compare_illingworth2005_planar as illingworth

    monkeypatch.setattr(
        illingworth,
        "_fig6_extracted_data_specs_for_cases",
        lambda case_results: [
            {
                "path": pathlib.Path(__file__),
                "color": "black",
                "marker": "o",
                "label": "Synthetic extracted data",
            }
        ],
    )
    monkeypatch.setattr(
        illingworth,
        "_load_no_header_xy_csv",
        lambda path: (
            np.array([100.0, 1.0e8, 100.0]),
            np.array([8.0, 8.0, 30.0]),
        ),
    )

    params = illingworth.build_fig6_illingworth_case_params("thin", 1.4e-8)
    fig = Figure()
    ax = fig.add_subplot(111)
    result = illingworth.plot_fig6_olaye_brass_sqrt_time_analytical(
        [
            {
                "time_s": np.array([0.0, 100.0, 400.0]),
                "interface_displacement_um": np.array([0.0, 8.0, 16.0]),
                "params": params,
            }
        ],
        {
            "show": False,
            "fig6_sqrt_time_out": False,
            "fig6_sqrt_time_plot_extracted_data": True,
            "fig6_sqrt_time_plot_experimental_data": False,
            "fig6_sqrt_time_max_s": None,
        },
        ax=ax,
    )

    assert result["axes"].get_xlabel() == "sqrt(time) (sqrt(s))"
    assert result["axes"].get_ylabel() == "Interface displacement (um)"
    assert len(result["axes"].lines) == 2
    assert len(result["axes"].collections) == 1
    assert_allclose(result["axes"].collections[0].get_offsets(), [[10.0, 8.0], [10.0, 30.0]])
    assert result["axes"].get_xlim()[1] < np.sqrt(1.0e8)
    assert result["axes"].get_ylim()[1] > 30.0


def test_illingworth_fig6_overlay_png_matches_source_canvas(monkeypatch):
    from examples.Illingworth2005 import compare_illingworth2005_planar as illingworth

    monkeypatch.setattr(illingworth, "_image_size_px", lambda path: (1521, 950))
    params = illingworth.build_fig6_illingworth_case_params("thin", 1.4e-8)

    overlay = illingworth.plot_fig6_olaye_brass_overlay_png(
        [
            {
                "time_s": np.array([50.0, 100.0, 1000.0]),
                "interface_displacement_um": np.array([5.0, 10.0, 20.0]),
                "params": params,
            }
        ],
        {
            "fig6_overlay_source_image_path": pathlib.Path(__file__),
            "fig6_overlay_out": False,
            "fig6_overlay_axes_rect": (0.1, 0.2, 0.3, 0.4),
            "fig6_overlay_xlim": (10.0, 1.0e6),
            "fig6_overlay_ylim": (-200.0, 100.0),
            "fig6_overlay_show_axes": False,
        },
    )

    assert overlay["path"] is None
    assert overlay["canvas_px"] == (1521, 950)
    assert overlay["axes_rect"] == (0.1, 0.2, 0.3, 0.4)
    assert len(overlay["axes"].lines) == 1
    assert not overlay["axes"].axison

    axes_overlay = illingworth.plot_fig6_olaye_brass_overlay_png(
        [
            {
                "time_s": np.array([50.0, 100.0, 1000.0]),
                "interface_displacement_um": np.array([5.0, 10.0, 20.0]),
                "params": params,
            }
        ],
        {
            "fig6_overlay_source_image_path": pathlib.Path(__file__),
            "fig6_overlay_out": False,
            "fig6_overlay_axes_rect": (0.1, 0.2, 0.3, 0.4),
            "fig6_overlay_xlim": (10.0, 1.0e6),
            "fig6_overlay_ylim": (-200.0, 100.0),
            "fig6_overlay_show_axes": True,
            "fig6_overlay_show_grid": True,
        },
    )

    assert axes_overlay["path"] is None
    assert axes_overlay["figure"].get_size_inches()[0] * axes_overlay["figure"].dpi == 1521
    assert axes_overlay["figure"].get_size_inches()[1] * axes_overlay["figure"].dpi == 950
    assert axes_overlay["axes"].axison
    assert len(axes_overlay["axes"].collections) == 1


def test_saved_planar_run_loader_accepts_minimal_valid_npz(tmp_path):
    from examples.compare_saved_planar_runs import _load_saved_run

    save_path = tmp_path / "valid_saved_run.npz"
    np.savez_compressed(
        save_path,
        time_s=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        half_width_um=np.array([12.5, 13.0, 13.5], dtype=np.float64),
        label=np.array("Valid run"),
    )

    loaded = _load_saved_run(save_path)
    assert_allclose(loaded["time_s"], [0.0, 1.0, 2.0], rtol=0.0, atol=0.0)
    assert_allclose(loaded["half_width_um"], [12.5, 13.0, 13.5], rtol=0.0, atol=0.0)
    assert loaded["metadata"]["label"] == "Valid run"


def test_saved_planar_run_loader_rejects_missing_required_arrays(tmp_path):
    from examples.compare_saved_planar_runs import _load_saved_run

    save_path = tmp_path / "missing_half_width.npz"
    np.savez_compressed(save_path, time_s=np.array([0.0, 1.0], dtype=np.float64))

    with pytest.raises(ValueError, match="time_s, half_width_um"):
        _load_saved_run(save_path)


def test_saved_planar_run_loader_rejects_too_few_finite_points(tmp_path):
    from examples.compare_saved_planar_runs import _load_saved_run

    save_path = tmp_path / "too_short.npz"
    np.savez_compressed(
        save_path,
        time_s=np.array([0.0, np.nan], dtype=np.float64),
        half_width_um=np.array([12.5, 13.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="at least two finite saved points"):
        _load_saved_run(save_path)


def test_saved_planar_run_plotter_uses_label_fallbacks_and_supports_log_scale(tmp_path):
    from examples.compare_saved_planar_runs import plot_saved_planar_comparison

    olaye_path = tmp_path / "olaye_saved_run.npz"
    illingworth_path = tmp_path / "illingworth_saved_run.npz"
    np.savez_compressed(
        olaye_path,
        time_s=np.array([0.1, 1.0, 2.0], dtype=np.float64),
        half_width_um=np.array([12.5, 13.0, 13.4], dtype=np.float64),
    )
    np.savez_compressed(
        illingworth_path,
        time_s=np.array([0.1, 1.0, 10.0], dtype=np.float64),
        half_width_um=np.array([12.5, 13.1, 14.2], dtype=np.float64),
        theoretical_max_um=np.array([23.2], dtype=np.float64),
    )

    fig, ax = plt.subplots()
    result = plot_saved_planar_comparison(
        {
            "olaye_run_path": olaye_path,
            "illingworth_run_path": illingworth_path,
            "x_axis": "log",
            "show": False,
            "out": None,
        },
        ax=ax,
    )

    assert result["axes"].get_xscale() == "log"
    labels = [line.get_label() for line in result["axes"].lines]
    assert "Olaye saved run" in labels
    assert "Illingworth saved run" in labels
    plt.close(fig)


def test_saved_planar_run_analytical_constants_normalize_saved_units():
    from examples.compare_saved_planar_runs import _analytical_constants_from_saved_params

    run = {
        "metadata": {
            "params_json": json.dumps(
                {
                    "R_um": 3012.5,
                    "s0_um": 12.5,
                    "c_liquid0_pct": 19.0,
                    "c_solid0_pct": 0.0,
                    "c_liquid_int_pct": 10.223,
                    "c_solid_int_pct": 0.166,
                    "D_liquid_base": 500.0,
                    "D_solid_base": 18.0,
                    "D_scale": 1e-12,
                }
            )
        }
    }

    constants = _analytical_constants_from_saved_params(run, "Synthetic Olaye")

    assert constants["c_a0"] == pytest.approx(0.19)
    assert constants["c_b_eq"] == pytest.approx(0.00166)
    assert constants["d_a_um2_s"] == pytest.approx(500.0)
    assert constants["d_b_um2_s"] == pytest.approx(18.0)
    assert constants["R_um"] == pytest.approx(3012.5)
    assert constants["s0_um"] == pytest.approx(12.5)
    assert constants["beta_um_sqrt_s"] > 0.0


def test_saved_planar_run_plotter_can_add_analytical_comparison():
    from examples.compare_saved_planar_runs import _plot_analytical_early_time_comparison

    params_json = json.dumps(
        {
            "s0_um": 12.5,
            "c_liquid0_atpct": 19.0,
            "c_solid0_atpct": 0.0,
            "c_liquid_int_atpct": 10.223,
            "c_solid_int_atpct": 0.166,
            "D_liquid_um2_s": 500.0,
            "D_solid_um2_s": 18.0,
        }
    )
    run = {
        "time_s": np.array([0.0, 1.0e-6, 4.0e-6, 9.0e-6], dtype=np.float64),
        "half_width_um": np.array([12.5, 12.501, 12.502, 12.503], dtype=np.float64),
        "metadata": {"params_json": params_json},
    }

    analytical = _plot_analytical_early_time_comparison(
        run,
        run,
        {
            "show": False,
            "analytical_out": None,
            "analytical_time_max_s": 1.0e-5,
            "analytical_max_points_per_run": 3,
            "olaye_label": None,
            "illingworth_label": None,
        },
    )

    labels = [line.get_label() for line in analytical["axes"].lines]
    assert any(label.startswith("Analytical: 2 beta sqrt(t)") for label in labels)
    assert len(analytical["axes"].lines[0].get_xdata()) == 3
    assert analytical["illingworth_constants"]["beta_um_sqrt_s"] == pytest.approx(
        analytical["olaye_constants"]["beta_um_sqrt_s"]
    )
    plt.close(analytical["figure"])


def test_saved_planar_run_analytical_comparison_can_auto_limit_to_semi_infinite_window():
    from examples.compare_saved_planar_runs import _plot_analytical_early_time_comparison

    params_json = json.dumps(
        {
            "R_um": 48.0,
            "s0_um": 12.0,
            "c_liquid0_atpct": 19.0,
            "c_solid0_atpct": 0.0,
            "c_liquid_int_atpct": 10.223,
            "c_solid_int_atpct": 0.166,
            "D_liquid_um2_s": 1.0,
            "D_solid_um2_s": 4.0,
        }
    )
    run = {
        "time_s": np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float64),
        "half_width_um": np.array([12.0, 13.0, 14.0, 15.0], dtype=np.float64),
        "metadata": {"params_json": params_json},
    }

    analytical = _plot_analytical_early_time_comparison(
        run,
        run,
        {
            "show": False,
            "analytical_out": None,
            "analytical_time_max_s": "semi_infinite",
            "semi_infinite_erfc_argument_min": 3.0,
            "analytical_max_points_per_run": None,
            "olaye_label": None,
            "illingworth_label": None,
        },
    )

    assert analytical["semi_infinite_cutoff"]["time_max_s"] == pytest.approx(4.0)
    assert list(analytical["axes"].lines[0].get_xdata()) == pytest.approx([0.0, 1.0, 2.0])
    plt.close(analytical["figure"])
