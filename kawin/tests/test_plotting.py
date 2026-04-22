from matplotlib.path import Path
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import optimize

from kawin.thermo import BinaryThermodynamics, MulticomponentThermodynamics, GeneralThermodynamics
from kawin.precipitation import PrecipitateModel, MatrixParameters, PrecipitateParameters
from kawin.diffusion import SinglePhaseModel
from kawin.diffusion.mesh import Cartesian1D, CartesianFD1D, Cartesian2D, StepProfile1D, BoundedRectangleProfile, ProfileBuilder
from kawin.diffusion import MovingBoundary1DModel, MovingBoundaryFD1DModel
from kawin.diffusion.mesh.MovingBoundary1D import debug_moving_boundary_state
from kawin.diffusion.mesh.MovingBoundaryFD1D import debug_moving_boundary_fd_state

#from kawin.precipitation.Plot import plotEuler
from kawin.precipitation.Plot import plotPrecipitateResults
from kawin.diffusion.Plot import plot1D, plot1DFlux, plot1DPhases, plot1DTwoAxis, plot2D, plot2DFluxes, plot2DPhases, plotMovingBoundaryState
from kawin.solver import explicitEulerIterator

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


def test_moving_boundary_fdm_state_plot_and_summary():
    profile = ProfileBuilder([(StepProfile1D(0, 0.2, 0.8), 'CR')])
    mesh = CartesianFD1D(['CR'], [-1, 1], 9)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        ['FCC_A1', 'BCC_A2'],
        {'FCC_A1': 1e-15, 'BCC_A2': 2e-15},
        (0.3, 0.7),
    )
    model = MovingBoundaryFD1DModel(
        mesh,
        ['NI', 'CR'],
        ['FCC_A1', 'BCC_A2'],
        therm,
        1000,
        interfacePosition=0.15,
        fluxGradientMode='post_diffusion',
        pstar=0.5,
    )

    fig, ax = plt.subplots()
    plotMovingBoundaryState(model, ax=ax)
    assert len(ax.lines) >= 2
    assert len(ax.collections) >= 2
    plt.close(fig)

    fig, ax = plt.subplots()
    model.plotMeshState(ax=ax)
    plt.close(fig)

    summary = model.describeMeshState(window=1)
    assert 'interface_position' in summary
    assert '<left of interface>' in summary
    assert '<right of interface>' in summary
    assert '<ignored>' in summary

    mesh.interface_position = 0.15
    mesh.pstar = 0.5
    summary2, _ = debug_moving_boundary_fd_state(
        mesh,
        composition=mesh.y[:, 0],
        plot=False,
        print_summary=False,
        window=1,
    )
    assert 'interface_position' in summary2


def test_moving_boundary_fdm_analytic_comparison_plot():
    def equation_a11(beta, c_a0, c_b0, c_a_eq, c_b_eq, d_a, d_b):
        return (
            ((c_a_eq - c_b_eq) * beta * np.sqrt(np.pi))
            - ((np.sqrt(d_a) * (c_a0 - c_a_eq)) / (1 + math.erf(beta / np.sqrt(d_a)))) * np.exp(-(beta**2) / d_a)
            + ((np.sqrt(d_b) * (c_b_eq - c_b0)) / (1 - math.erf(beta / np.sqrt(d_b)))) * np.exp(-(beta**2) / d_b)
        )

    def solve_beta(c_a0, c_b0, c_a_eq, c_b_eq, d_a, d_b):
        grid = np.linspace(-10.0, 10.0, 4001)
        values = np.array([equation_a11(x, c_a0, c_b0, c_a_eq, c_b_eq, d_a, d_b) for x in grid], dtype=np.float64)
        for x_left, x_right, y_left, y_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
            if not (np.isfinite(y_left) and np.isfinite(y_right)):
                continue
            if y_left == 0:
                return float(x_left)
            if np.sign(y_left) != np.sign(y_right):
                sol = optimize.root_scalar(
                    equation_a11,
                    bracket=[float(x_left), float(x_right)],
                    method='brentq',
                    args=(c_a0, c_b0, c_a_eq, c_b_eq, d_a, d_b),
                    rtol=1e-14,
                    xtol=1e-14,
                )
                return float(sol.root)
        raise ValueError("Could not bracket an analytic moving-boundary root.")

    # L=1
    # N=100
    # interface_position = 0.5125
    # c_a_eq = 0.35
    # c_b_eq = 0.65
    # c_a0 = 0.1
    # c_b0 = 0.8
    # d_a = 1.0
    # d_b = 10.0
    # t_end = 0.002

    L=374.5+190.5
    N=100
    interface_position = 374.5
    c_a_eq = 0.325
    c_b_eq = 0.369
    c_a0 = 0.291
    c_b0 = 0.394
    d_a = 5.0
    d_b = 100.0
    t_end = 1e4
    
    record_input=10
    vIt_input=5000
    verbose_input = False if record_input == False else True
    pre=False
    import time
    t0_overall=time.perf_counter()
    profile = ProfileBuilder([(StepProfile1D(interface_position, c_a0, c_b0), 'CR')])
    mesh = CartesianFD1D(['CR'], [0, L], N+1)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        ['ALPHA', 'BETA'],
        {'ALPHA': d_a, 'BETA': d_b},
        (c_a_eq, c_b_eq),
    )

    # def profTest(t):
    #     post_model_basic = MovingBoundaryFD1DModel(
    #         mesh,
    #         ['FE', 'CR'],
    #         ['ALPHA', 'BETA'],
    #         therm,
    #         temperature=1000,
    #         interfacePosition=interface_position,
    #         fluxGradientMode='post_diffusion',
    #         interfaceUpdate='basic',
    #         pstar=0.5,
    #         record=record_input,
    #     )
    #     post_model_basic.solve(t, iterator=explicitEulerIterator, vIt=vIt_input, verbose=verbose_input)
    #     return post_model_basic

    # import cProfile
    # import pstats

    # pr = cProfile.Profile()
    # pr.runcall(profTest, 1e4)
    # pr.create_stats()

    # p = pstats.Stats(pr)
    # p.strip_dirs()
    # p.sort_stats('cumtime').print_stats(40)
    # p.sort_stats('tottime').print_stats(40)
    
    # pr.dump_stats("mbfdm_profile.pstats")
    # assert (True==False)
    # import subprocess
    # subprocess.run("& 'C:\ProgramData\anaconda3\envs\kawin\python.exe' -m gprof2dot -f pstats mbfdm_profile.pstats -o mbfdm_profile.dot", shell=True, check=True)
    # subprocess.run("gprof2dot -f pstats mbfdm_profile.pstats | dot -Tpng -o output.png", shell=True, check=True)



    post_model_basic = MovingBoundaryFD1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        therm,
        temperature=1000,
        interfacePosition=interface_position,
        fluxGradientMode='post_diffusion',
        interfaceUpdate='basic',
        pstar=0.5,
        record=record_input,
    )
    t0_post_basic_solve=time.perf_counter()
    post_model_basic.solve(t_end, iterator=explicitEulerIterator, vIt=vIt_input, verbose=verbose_input)
    t1_post_basic_solve=time.perf_counter()
    print(f"Post-Diffusion Basic Solve Time: {t1_post_basic_solve - t0_post_basic_solve}")
    if pre==True:
        pre_model_basic = MovingBoundaryFD1DModel(
            mesh,
            ['FE', 'CR'],
            ['ALPHA', 'BETA'],
            therm,
            temperature=1000,
            interfacePosition=interface_position,
            fluxGradientMode='pre_diffusion',
            interfaceUpdate='basic',
            pstar=0.5,
            record=record_input,
        )
        pre_model_basic.solve(t_end, iterator=explicitEulerIterator, vIt=vIt_input, verbose=verbose_input)

    post_model_corr = MovingBoundaryFD1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        therm,
        temperature=1000,
        interfacePosition=interface_position,
        fluxGradientMode='post_diffusion',
        interfaceUpdate='lee_oh_corrected',
        pstar=0.5,
        record=record_input,
    )
    t0_post_corr_solve=time.perf_counter()
    post_model_corr.solve(t_end, iterator=explicitEulerIterator, vIt=vIt_input, verbose=verbose_input)
    t1_post_corr_solve=time.perf_counter()
    print(f"Post-Diffusion Corrected Solve Time: {t1_post_corr_solve - t0_post_corr_solve}")
    if pre==True:
        pre_model_corr = MovingBoundaryFD1DModel(
            mesh,
            ['FE', 'CR'],
            ['ALPHA', 'BETA'],
            therm,
            temperature=1000,
            interfacePosition=interface_position,
            fluxGradientMode='pre_diffusion',
            interfaceUpdate='lee_oh_corrected',
            pstar=0.5,
            record=record_input,
        )
        pre_model_corr.solve(t_end, iterator=explicitEulerIterator, vIt=vIt_input, verbose=verbose_input)
    t1_overall=time.perf_counter()
    print(f"Overall Time: {t1_overall - t0_overall}")

    t_post_basic = np.array(post_model_basic.interfaceData._time[:post_model_basic.interfaceData.N+1], dtype=np.float64)
    s_post_basic = np.array(post_model_basic.interfaceData._y[:post_model_basic.interfaceData.N+1], dtype=np.float64)
    sqrt_t_post_basic = np.sqrt(t_post_basic)
    delta_s_post_basic = s_post_basic - s_post_basic[0]
    
    if pre==True:
        t_pre_basic = np.array(pre_model_basic.interfaceData._time[:pre_model_basic.interfaceData.N+1], dtype=np.float64)
        s_pre_basic = np.array(pre_model_basic.interfaceData._y[:pre_model_basic.interfaceData.N+1], dtype=np.float64)
        sqrt_t_pre_basic = np.sqrt(t_pre_basic)
        delta_s_pre_basic = s_pre_basic - s_pre_basic[0]
    
    t_post_corr = np.array(post_model_corr.interfaceData._time[:post_model_corr.interfaceData.N+1], dtype=np.float64)
    s_post_corr = np.array(post_model_corr.interfaceData._y[:post_model_corr.interfaceData.N+1], dtype=np.float64)
    sqrt_t_post_corr = np.sqrt(t_post_corr)
    delta_s_post_corr = s_post_corr - s_post_corr[0]

    if pre==True:
        t_pre_corr = np.array(pre_model_corr.interfaceData._time[:pre_model_corr.interfaceData.N+1], dtype=np.float64)
        s_pre_corr = np.array(pre_model_corr.interfaceData._y[:pre_model_corr.interfaceData.N+1], dtype=np.float64)
        sqrt_t_pre_corr = np.sqrt(t_pre_corr)
        delta_s_pre_corr = s_pre_corr - s_pre_corr[0]

    beta = solve_beta(c_a0, c_b0, c_a_eq, c_b_eq, d_a, d_b)
    analytic_delta_s = lambda sqrt_t: 2.0 * beta * sqrt_t

    if pre==True:
        assert post_model_basic.mesh.N-1 == pre_model_basic.mesh.N-1 == post_model_corr.mesh.N-1 == pre_model_corr.mesh.N-1 == N
    else: 
        assert post_model_basic.mesh.N-1 == post_model_corr.mesh.N-1 == N 
    num_points_post_basic = max(5, min(len(sqrt_t_post_basic), math.floor(min(interface_position, L - interface_position) / (L / N)) - 5))
    sqrt_t_post_basic_sub = sqrt_t_post_basic[:num_points_post_basic]
    delta_s_post_basic_sub = delta_s_post_basic[:num_points_post_basic]
    analytic_post_basic_sub = analytic_delta_s(sqrt_t_post_basic_sub)[:num_points_post_basic]

    if pre==True:
        num_points_pre_basic = max(5, min(len(sqrt_t_pre_basic), math.floor(min(interface_position, L - interface_position) / (L / N)) - 5))
        sqrt_t_pre_basic_sub = sqrt_t_pre_basic[:num_points_pre_basic]
        delta_s_pre_basic_sub = delta_s_pre_basic[:num_points_pre_basic]
        analytic_pre_basic_sub = analytic_delta_s(sqrt_t_pre_basic_sub)[:num_points_pre_basic]

    num_points_post_corr = max(5, min(len(sqrt_t_post_corr), math.floor(min(interface_position, L - interface_position) / (L / N)) - 5))
    sqrt_t_post_corr_sub = sqrt_t_post_corr[:num_points_post_corr]
    delta_s_post_corr_sub = delta_s_post_corr[:num_points_post_corr]
    analytic_post_corr_sub = analytic_delta_s(sqrt_t_post_corr_sub)[:num_points_post_corr]

    if pre==True:
        num_points_pre_corr = max(5, min(len(sqrt_t_pre_corr), math.floor(min(interface_position, L - interface_position) / (L / N)) - 5))
        sqrt_t_pre_corr_sub = sqrt_t_pre_corr[:num_points_pre_corr]
        delta_s_pre_corr_sub = delta_s_pre_corr[:num_points_pre_corr]
        analytic_pre_corr_sub = analytic_delta_s(sqrt_t_pre_corr_sub)[:num_points_pre_corr]

    if (np.ravel(post_model_corr.data.y(t_post_corr[num_points_post_corr-1]))[[0,-1]]==np.array([c_a0, c_b0])).all()!=True:
        np.ravel(post_model_corr.data.y(t_post_corr[num_points_post_corr-1]))
        print("WARNING: SEMIINFINITE ASSUMPTION MAY BE VIOLATED")

    fig, ax = plt.subplots()
    ax.plot(sqrt_t_post_basic_sub, delta_s_post_basic_sub, 'o', label='MovingBoundaryFD1D Post-Diffusion Basic', color='tab:blue')
    ax.plot(sqrt_t_post_corr_sub, delta_s_post_corr_sub, 'o', label='MovingBoundaryFD1D Post-Diffusion Corrected', color='tab:green')
    if pre==True:
        ax.plot(sqrt_t_pre_basic_sub, delta_s_pre_basic_sub, 'x', label='MovingBoundaryFD1D Pre-Diffusion Basic', color='tab:orange')
        ax.plot(sqrt_t_pre_corr_sub, delta_s_pre_corr_sub, 'x', label='MovingBoundaryFD1D Pre-Diffusion Corrected', color='tab:red')
    ax.plot(sqrt_t_post_corr_sub, analytic_post_corr_sub, label='Analytic', color='k')
    parameter_text = (
        f"L = {L}\n"
        f"N = {N}\n"
        f"interface_position = {interface_position}\n"
        f"c_a0 = {c_a0}\n"
        f"c_b0 = {c_b0}\n"
        f"c_a_eq = {c_a_eq}\n"
        f"c_b_eq = {c_b_eq}\n"
        f"d_a = {d_a}\n"
        f"d_b = {d_b}"
    )
    ax.text(
        0.02,
        0.25,
        parameter_text,
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=9,
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.85, 'edgecolor': '0.7'},
    )
    ax.set_xlabel('sqrt(t)')
    ax.set_ylabel('Change in interface position')
    # ax.set_ylim(-0.007, 0.0015)
    # ax.set_xlim(0, 0.012)
    ax.legend()
    plt.show()
    
    idealized_mass_funcOfStartingInterfacePosition = lambda startingInterfacePosition: (c_a0 * startingInterfacePosition) + (c_b0 * (L - startingInterfacePosition))
    idealized_mass = idealized_mass_funcOfStartingInterfacePosition(interface_position)
    
    fig2, ax2 = plt.subplots()
    post_basic_posAndMassErrors_arr = np.array([(((s_post_basic[i]-s_post_basic[0])-2*beta*np.sqrt(t))/(2*beta*np.sqrt(t)), (post_model_basic.getTotalMass(time=t)-post_model_basic._initialInventory)/post_model_basic._initialInventory) for i, t in enumerate(t_post_basic)])[:num_points_post_basic]
    post_corr_posAndMassErrors_arr = np.array([(((s_post_corr[i]-s_post_corr[0])-2*beta*np.sqrt(t))/(2*beta*np.sqrt(t)), (post_model_corr.getTotalMass(time=t)-post_model_corr._initialInventory)/post_model_corr._initialInventory) for i, t in enumerate(t_post_corr)])[:num_points_post_corr]
    if pre==True:
        pre_basic_posAndMassErrors_arr = np.array([(((s_pre_basic[i]-s_pre_basic[0])-2*beta*np.sqrt(t))/(2*beta*np.sqrt(t)), (pre_model_basic.getTotalMass(time=t)-pre_model_basic._initialInventory)/pre_model_basic._initialInventory) for i, t in enumerate(t_pre_basic)])[:num_points_pre_basic]
        pre_corr_posAndMassErrors_arr = np.array([(((s_pre_corr[i]-s_pre_corr[0])-2*beta*np.sqrt(t))/(2*beta*np.sqrt(t)), (pre_model_corr.getTotalMass(time=t)-pre_model_corr._initialInventory)/pre_model_corr._initialInventory) for i, t in enumerate(t_pre_corr)])[:num_points_pre_corr]

    ax2.plot(post_basic_posAndMassErrors_arr[:,1], post_basic_posAndMassErrors_arr[:,0], 'o-', fillstyle='none', label='Post-Diffusion Basic', color='tab:blue')
    ax2.plot(post_corr_posAndMassErrors_arr[:,1], post_corr_posAndMassErrors_arr[:,0], 'o-', fillstyle='none', label='Post-Diffusion Corrected', color='tab:green')
    if pre==True:
        ax2.plot(pre_basic_posAndMassErrors_arr[:,1], pre_basic_posAndMassErrors_arr[:,0], 'x-', fillstyle='none', label='Pre-Diffusion Basic', color='tab:orange')
        ax2.plot(pre_corr_posAndMassErrors_arr[:,1], pre_corr_posAndMassErrors_arr[:,0], 'x-', fillstyle='none', label='Pre-Diffusion Corrected', color='tab:red')
    ax2.scatter(post_basic_posAndMassErrors_arr[:,1][1], post_basic_posAndMassErrors_arr[:,0][1], marker='s', color='tab:blue')
    ax2.scatter(post_corr_posAndMassErrors_arr[:,1][1], post_corr_posAndMassErrors_arr[:,0][1], marker='s', color='tab:green')
    if pre==True:
        ax2.scatter(pre_basic_posAndMassErrors_arr[:,1][1], pre_basic_posAndMassErrors_arr[:,0][1], marker='s', color='tab:orange')
        ax2.scatter(pre_corr_posAndMassErrors_arr[:,1][1], pre_corr_posAndMassErrors_arr[:,0][1], marker='s', color='tab:red')
    ax2.hlines(0, ax2.get_xlim()[0], ax2.get_xlim()[1], colors='gray', linewidth=1, alpha=0.4, zorder=-1)
    ax2.vlines(0, ax2.get_ylim()[0], ax2.get_ylim()[1], colors='gray', linewidth=1, alpha=0.4, zorder=-1)
    ax2.set_xlabel('Mass Error (relative to initial inventory)', fontsize=12)
    ax2.set_ylabel('Interface Displacement Error (relative to analytic solution)', fontsize=12)
    ax2.legend()

    print(post_basic_posAndMassErrors_arr[-1])
    if pre==True:
        print(pre_basic_posAndMassErrors_arr[-1])
    print(post_corr_posAndMassErrors_arr[-1])
    if pre==True:
        print(pre_corr_posAndMassErrors_arr[-1])

    print((post_model_basic._initialInventory, post_model_basic.getTotalMass(), post_model_basic.getTotalMass()-post_model_basic._initialInventory))
    if pre==True:
        print((pre_model_basic._initialInventory, pre_model_basic.getTotalMass(), pre_model_basic.getTotalMass()-pre_model_basic._initialInventory))
    print((post_model_corr._initialInventory, post_model_corr.getTotalMass(), post_model_corr.getTotalMass()-post_model_corr._initialInventory))
    if pre==True:
        print((pre_model_corr._initialInventory, pre_model_corr.getTotalMass(), pre_model_corr.getTotalMass()-pre_model_corr._initialInventory))

    def initial_mass_funcOfStartingInterfacePosition(startingInterfacePosition):
        
        profile = ProfileBuilder([(StepProfile1D(startingInterfacePosition, c_a0, c_b0), 'CR')])
        mesh = CartesianFD1D(['CR'], [0, L], N+1)
        mesh.setResponseProfile(profile)
        therm = ConstantBinaryThermodynamics(
            ['ALPHA', 'BETA'],
            {'ALPHA': d_a, 'BETA': d_b},
            (c_a_eq, c_b_eq),
        )
        model_for_initMass = MovingBoundaryFD1DModel(
            mesh,
            ['FE', 'CR'],
            ['ALPHA', 'BETA'],
            therm,
            temperature=1000,
            interfacePosition=startingInterfacePosition,
            fluxGradientMode='post_diffusion',
            interfaceUpdate='basic',
            pstar=0.5,
            record=True,
        )
        return model_for_initMass._initialInventory

    initial_mass_funcOfStartingInterfacePosition(interface_position)-post_model_basic._initialInventory
    
    startingMass_diff = lambda startingInterfacePosition: initial_mass_funcOfStartingInterfacePosition(startingInterfacePosition)-idealized_mass_funcOfStartingInterfacePosition(interface_position)
    sol = optimize.root_scalar(
                    startingMass_diff,
                    bracket=[float(0)+((3/N)*L), float(L)-((3/N)*L)],
                    method='brentq',
                    rtol=1e-14,
                    xtol=1e-14,
                )

    
    fig3, ax3 = plt.subplots()
    startingPosition_arr = np.linspace(interface_position-((L/N)*2), interface_position+((L/N)*2), 60)+1e-8
    ax3.plot(startingPosition_arr, [initial_mass_funcOfStartingInterfacePosition(pos) for pos in startingPosition_arr], 'o-')
    ax3.plot(startingPosition_arr, [idealized_mass_funcOfStartingInterfacePosition(pos) for pos in startingPosition_arr], 'o-')
    ax3.hlines(idealized_mass, startingPosition_arr[0], startingPosition_arr[-1], colors='k', linestyles='dashed', label='Idealized Mass')
    ax3.vlines(interface_position, initial_mass_funcOfStartingInterfacePosition(startingPosition_arr[0]), initial_mass_funcOfStartingInterfacePosition(startingPosition_arr[-1]), colors='k', linestyles='dashed', label='Idealized Mass')

    import pandas as pd
    fig8_presentMethod_df = pd.read_csv(r'C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\kawin\examples\leeAndOh1996_data\fig8_presentMethodCurve.csv', names=['t', 'normalized_thickness'])
    fig8_presentMethod_df = fig8_presentMethod_df.sort_values(by=['t'])
    fig4, ax4 = plt.subplots()
    fig8_presentMethod_df.plot(x='t', y='normalized_thickness', ax=ax4, label='Lee and Oh 1996 Fig. 8 Present Method Curve', color='k')
    ax4.plot(t_post_basic, (L-s_post_basic)/(L-s_post_basic[0]), 'o-', fillstyle='none', label='Post-Diffusion Basic', color='tab:blue')
    if pre==True:
        ax4.plot(t_pre_basic, (L-s_pre_basic)/(L-s_pre_basic[0]), 'x-', fillstyle='none', label='Pre-Diffusion Basic', color='tab:orange')
    ax4.plot(t_post_corr, (L-s_post_corr)/(L-s_post_corr[0]), 'o-', fillstyle='none', label='Post-Diffusion Corrected', color='tab:green')
    if pre==True:
        ax4.plot(t_pre_corr, (L-s_pre_corr)/(L-s_pre_corr[0]), 'x-', fillstyle='none', label='Pre-Diffusion Corrected', color='tab:red')

    figure8_presentMethod_oldExtract_df = pd.read_csv(r"C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\finiteDifference\LeeAndOh_figure8_presentMethod.csv", low_memory=False, names=['x', 'y'])
    figure8_presentMethod_oldExtract_df = figure8_presentMethod_oldExtract_df.sort_values(by=['x'])
    figure8_presentMethod_oldExtract_df.plot(x='x', y='y', ax=ax4, label='Lee and Oh 1996 Fig. 8 Present Method Curve (OLD EXTRACTION)', color='dimgray', zorder=-1)

    ax4.legend()
    ax4.set_xscale('log')

    ''' Quantitative Comparison of difference in t-s curves '''
    lstOf_indx_TS_arrs_dict = [
        {'t':t_post_basic.copy(), 's':s_post_basic.copy(), 'name':'post_basic'},
        {'t':t_post_corr.copy(), 's':s_post_corr.copy(), 'name':'post_corr'},
    ]
    TS_df = pd.DataFrame(lstOf_indx_TS_arrs_dict)
    TS_df['s_norm'] = TS_df['s'].apply(lambda s_arr: (L-s_arr)/(L-s_arr[0]))
    from scipy.interpolate import splrep, splev
    def buildAndEvalSpline(x, y, xToEval):
        spl = splrep(x, y, k=3)
        return splev(xToEval, spl, ext=2)

    T_toEvalAt_1 = np.logspace(-1, -2, 1000)-1e-10
    T_toEvalAt_2 = np.logspace(-2, np.log10(t_end), 1000)-1e-10
    TS_df['sNorm_splEval'] = TS_df.apply(lambda r: buildAndEvalSpline(x=r['t'], y=r['s_norm'], xToEval=T_toEvalAt_1), axis=1)


    MSE_func = lambda arr1, arr2: np.linalg.norm(arr1-arr2)/len(arr1)

    currentExtract_splEval = buildAndEvalSpline(x=fig8_presentMethod_df['t'].to_numpy(), y=fig8_presentMethod_df['normalized_thickness'].to_numpy(), xToEval=np.logspace(np.log10(fig8_presentMethod_df['t'].min())+1e-6, np.log10(figure8_presentMethod_oldExtract_df['x'].max())-1e-6, 1000))
    oldExtract_splEval = buildAndEvalSpline(x=figure8_presentMethod_oldExtract_df['x'].to_numpy(), y=figure8_presentMethod_oldExtract_df['y'].to_numpy(), xToEval=np.logspace(np.log10(fig8_presentMethod_df['t'].min())+1e-6, np.log10(figure8_presentMethod_oldExtract_df['x'].max())-1e-6, 1000))
    print(f"MSE_func(currentExtract_splEval, oldExtract_splEval): {MSE_func(currentExtract_splEval, oldExtract_splEval)}")

    MSE_dict={}
    for indx1 in range(len(TS_df)):
        for indx2 in range(len(TS_df)):
            MSE_val = MSE_func(TS_df['sNorm_splEval'].iloc[indx1], TS_df['sNorm_splEval'].iloc[indx2])
            MSE_dict.update({(indx1, indx2):MSE_val})
            
    MSE_df = pd.DataFrame(columns=range(len(TS_df)), index=range(len(TS_df)))
    for MSE_key, MSE_val in MSE_dict.items():
        MSE_df.loc[MSE_key[0], MSE_key[1]] = MSE_val
    MSE_df=MSE_df.astype(float)


    assert len(ax.lines) == 5
    assert len(ax.texts) >= 1
    assert np.all(np.isfinite(sqrt_t_post_basic_sub))
    assert np.all(np.isfinite(delta_s_post_basic_sub))
    assert np.all(np.isfinite(analytic_post_basic_sub))
    # plt.close(fig)
