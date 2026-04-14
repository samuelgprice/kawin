import numpy as np
import matplotlib.pyplot as plt

from kawin.thermo import BinaryThermodynamics, MulticomponentThermodynamics, GeneralThermodynamics
from kawin.precipitation import PrecipitateModel, MatrixParameters, PrecipitateParameters
from kawin.diffusion import SinglePhaseModel
from kawin.diffusion.mesh import Cartesian1D, CartesianFD1D, Cartesian2D, StepProfile1D, BoundedRectangleProfile, ProfileBuilder
from kawin.diffusion import MovingBoundary1DModel
from kawin.diffusion.mesh.MovingBoundary1D import debug_moving_boundary_state

#from kawin.precipitation.Plot import plotEuler
from kawin.precipitation.Plot import plotPrecipitateResults
from kawin.diffusion.Plot import plot1D, plot1DFlux, plot1DPhases, plot1DTwoAxis, plot2D, plot2DFluxes, plot2DPhases, plotMovingBoundaryState

from kawin.tests.datasets import NICRAL_TDB

binPrecTherm = BinaryThermodynamics(NICRAL_TDB, ['NI', 'AL'], ['FCC_A1', 'FCC_L12', 'C14_LAVES', 'C15_LAVES'], drivingForceMethod='tangent')
ternPrecTherm = MulticomponentThermodynamics(NICRAL_TDB, ['NI', 'AL', 'CR'], ['FCC_A1', 'FCC_L12', 'C14_LAVES', 'C15_LAVES'], drivingForceMethod='tangent')

binDiffTherm = GeneralThermodynamics(NICRAL_TDB, ['NI', 'CR'], ['FCC_A1', 'BCC_A2'])
ternDiffTherm = GeneralThermodynamics(NICRAL_TDB, ['NI', 'CR', 'AL'], ['FCC_A1', 'BCC_A2'])


class ConstantBinaryThermodynamics:
    def __init__(self, phases, diffusivities, interface_compositions):
        self.phases = phases
        self.diffusivities = diffusivities
        self.interface_compositions = interface_compositions

    def clearCache(self):
        return

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None):
        values = np.atleast_1d(T).astype(np.float64)
        return np.squeeze(np.ones(values.shape, dtype=np.float64) * self.diffusivities[phase])

    def getInterfacialComposition(self, T, gExtra=0, precPhase=None):
        values = np.atleast_1d(T).astype(np.float64)
        left = np.ones(values.shape, dtype=np.float64) * self.interface_compositions[0]
        right = np.ones(values.shape, dtype=np.float64) * self.interface_compositions[1]
        return np.squeeze(left), np.squeeze(right)

def test_precipitate_plotting():
    binary_matrix = MatrixParameters(['AL'])
    ternary_matrix = MatrixParameters(['AL', 'CR'])

    fcc_prec = PrecipitateParameters('FCC_L12')
    fcc_prec.gamma = 0.1

    c14_prec = PrecipitateParameters('C14_LAVES')
    c14_prec.gamma = 0.1

    c15_prec = PrecipitateParameters('C15_LAVES')
    c15_prec.gamma = 0.1

    temperature = 500
    binary_single = PrecipitateModel(binary_matrix, [fcc_prec], binPrecTherm, temperature)
    binary_multi = PrecipitateModel(binary_matrix, [fcc_prec, c14_prec, c15_prec], binPrecTherm, temperature)
    ternary_single = PrecipitateModel(ternary_matrix, [fcc_prec], ternPrecTherm, temperature)
    ternary_multi = PrecipitateModel(ternary_matrix, [fcc_prec, c14_prec, c15_prec], ternPrecTherm, temperature)

    models = [
        binary_single,
        binary_multi,
        ternary_single,
        ternary_multi,
    ]

    for m in models:
        varTypes = [
            ('volume fraction', {'phases': m.phases[0]}, 1),
            ('critical radius', {'phases': None}, len(m.phases)),
            ('average radius', {'phases': m.phases}, len(m.phases)),
            ('volume average radius', {'phases': ['total'] + [p for p in m.phases]}, len(m.phases)+1),
            ('aspect ratio', {'phases': None}, len(m.phases)),
            ('driving force', {'phases': [m.phases[0]]}, 1),
            ('nucleation rate', {'phases': ['total']}, 1),
            ('precipitate density', {'phases': ['total', m.phases[0]]}, 2),
            ('temperature', {}, 1),
            ('composition', {'elements': m.elements[0]}, 1),
            ('eq comp alpha', {'elements': None, 'phase': m.phases[0]}, len(m.elements)),
            ('eq comp beta', {'elements': m.elements, 'phase': m.phases[-1]}, len(m.elements)),
            ('supersaturation', {'phases': None}, len(m.phases)),
            ('eq volume fraction', {'phases': m.phases[0]}, 1),
            ('psd', {'phases': None}, len(m.phases)),
            ('pdf', {'phases': m.phases[0]}, 1),
            ('cdf', {'phases': m.phases}, len(m.phases)),
        ]
        for v in varTypes:
            print(v[0])
            fig, ax = plt.subplots(1,1)
            plotPrecipitateResults(m, v[0], ax=ax, **v[1])
            numLines = len(ax.lines)
            plt.close(fig)
            assert numLines == v[2]

def test_diffusion_plotting1d():
    #Single phase and Homogenizaton model goes through the same path for plotting
    for mesh_cls in (Cartesian1D, CartesianFD1D):
        profile_binary = ProfileBuilder([(StepProfile1D(0.5, 0.1, 0.9), 'CR')])
        mesh_binary = mesh_cls(['CR'], [-1,1], 100)
        mesh_binary.setResponseProfile(profile_binary)

        profile_ternary = ProfileBuilder([(StepProfile1D(0.5, [0.1,0.2], [0.9,0.01]), ['CR', 'AL'])])
        mesh_ternary = mesh_cls(['CR', 'AL'], [-1,1], 100)
        mesh_ternary.setResponseProfile(profile_ternary)

        temperature = 1000
        binary_single = SinglePhaseModel(mesh_binary, ['NI', 'CR'], ['FCC_A1'], binDiffTherm, temperature)
        binary_multi = SinglePhaseModel(mesh_binary, ['NI', 'CR'], ['FCC_A1', 'BCC_A2'], binDiffTherm, temperature)
        ternary_single = SinglePhaseModel(mesh_ternary, ['NI', 'CR', 'AL'], ['FCC_A1'], ternDiffTherm, temperature)
        ternary_multi = SinglePhaseModel(mesh_ternary, ['NI', 'CR', 'AL'], ['FCC_A1', 'BCC_A2'], ternDiffTherm, temperature)

        models = [
            (binary_single, 2, 1),
            (binary_multi, 2, 2),
            (ternary_single, 3, 1),
            (ternary_multi, 3, 2),
        ]

        for m in models:
            #For each plot, check that the number of lines correspond to number of elements or phases
            #For 'plot', number of lines should be elements (with or without reference) or a single element
            #For 'plotTwoAxis', number of lines for each axis should be length of input array
            #For 'plotPhases', number of lines is number of phases or single phase
            fig, ax = plt.subplots()
            plot1D(m[0], elements=m[0].allElements, ax=ax)
            assert len(ax.lines) == m[1]
            plt.close(fig)

            fig, ax = plt.subplots()
            plot1D(m[0], elements=None, ax=ax)
            assert len(ax.lines) == m[1]-1
            plt.close(fig)

            fig, ax = plt.subplots()
            plot1D(m[0], elements=m[0].allElements[0], ax=ax)
            assert len(ax.lines) == 1
            plt.close(fig)

            fig, axL = plt.subplots()
            axR = axL.twinx()
            plot1DTwoAxis(m[0], m[0].allElements[0], m[0].allElements[1:], axL=axL, axR=axR)
            assert len(axL.lines) == 1
            assert len(axR.lines) == len(m[0].allElements)-1
            plt.close(fig)

            fig, ax = plt.subplots()
            plot1DPhases(m[0], phases=None, ax=ax)
            assert len(ax.lines) == m[2]
            plt.close(fig)

            fig, ax = plt.subplots()
            plot1DFlux(m[0], elements=m[0].elements, ax=ax)
            assert len(ax.lines) == m[1]-1

def test_diffusion_plotting2d():
    '''
    Test that we can plot in 2d without error
    '''
    profile = ProfileBuilder()
    profile.addBuildStep(BoundedRectangleProfile([0,0], [1e-3, 1e-3], [0.5, 0.4], [0.1, 0.2]), ['CR', 'AL'])
    mesh = Cartesian2D(['CR', 'AL'], [-1e-3, 1e-3], 50, [-1e-3, 1e-3], 50)
    mesh.setResponseProfile(profile)

    model = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1', 'BCC_A2'], ternDiffTherm, 1000)

    fig, ax = plt.subplots()
    plot2D(model, 'CR', ax=ax)
    plt.close(fig)

    fig, ax = plt.subplots()
    plot2DPhases(model, 'FCC_A1', ax=ax)
    plt.close(fig)

    fig, ax = plt.subplots()
    plot2DFluxes(model, 'AL', 'x', ax=ax)
    plt.close(fig)

    fig, ax = plt.subplots()
    plot2DFluxes(model, 'CR', 'y', ax=ax)
    plt.close(fig)


def test_moving_boundary_state_plot_and_summary():
    profile = ProfileBuilder([(StepProfile1D(0, 0.2, 0.8), 'CR')])
    mesh = Cartesian1D(['CR'], [-1, 1], 8)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        ['FCC_A1', 'BCC_A2'],
        {'FCC_A1': 1e-15, 'BCC_A2': 2e-15},
        (0.3, 0.7),
    )
    model = MovingBoundary1DModel(
        mesh,
        ['NI', 'CR'],
        ['FCC_A1', 'BCC_A2'],
        therm,
        1000,
        interfacePosition=0.15,
    )

    fig, ax = plt.subplots()
    plotMovingBoundaryState(model, ax=ax)
    assert len(ax.lines) >= 3
    assert len(ax.collections) >= 2
    plt.close(fig)

    fig, ax = plt.subplots()
    model.plotMeshState(ax=ax)
    plt.close(fig)

    summary = model.describeMeshState(window=1)
    assert 'interface_position' in summary
    assert '<left of interface>' in summary
    assert '<right of interface>' in summary

    mesh.interface_position = 0.15
    summary2, _ = debug_moving_boundary_state(
        mesh,
        composition=mesh.y[:, 0],
        plot=False,
        print_summary=False,
        window=1,
    )
    assert 'interface_position' in summary2
