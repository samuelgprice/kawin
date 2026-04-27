#%%

# %matplotlib inline
%config InlineBackend.figure_format = 'svg'
from matplotlib.path import Path
import numpy as np
import matplotlib.pyplot as plt
import math
import hashlib
import json
from scipy import optimize
import pathlib

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


MODEL_CACHE_VERSION = "v1"


def _normalize_cache_value(value):
    '''
    Converts numpy/path-like values into JSON-serializable cache metadata
    '''
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _normalize_cache_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_cache_value(v) for v in value]
    return value


def build_model_cache_name(prefix, params):
    '''
    Builds a deterministic cache filename from a parameter dictionary
    '''
    normalized = _normalize_cache_value(params)
    payload = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]
    return f"{prefix}_{digest}.npz"


def register_cache_metadata(metadata_path, prefix, params):
    '''
    Records cache parameters for a generated cache hash in a common JSON file.
    '''
    normalized = _normalize_cache_value(params)
    payload = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]
    cache_filename = f"{prefix}_{digest}.npz"

    if metadata_path.exists():
        with metadata_path.open('r', encoding='utf-8') as fh:
            metadata = json.load(fh)
    else:
        metadata = {}

    metadata[cache_filename] = {
        'prefix': prefix,
        'hash': digest,
        'params': normalized,
    }

    with metadata_path.open('w', encoding='utf-8') as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)

    return cache_filename


def diffusion_constraints_to_cache_dict(constraints):
    '''
    Converts DiffusionConstraints into stable cache metadata.
    '''
    return {
        'minComposition': constraints.minComposition,
        'vonNeumannThreshold': constraints.vonNeumannThreshold,
        'maxCompositionChange': constraints.maxCompositionChange,
        'movingBoundaryThreshold': constraints.movingBoundaryThreshold,
        'movingBoundaryMassTolerance': constraints.movingBoundaryMassTolerance,
        'movingBoundaryMassAction': constraints.movingBoundaryMassAction,
    }

#%%

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

l_a = 374.5
l_b = 190.5
L= l_a + l_b
N=80
interface_position = l_a
c_a_eq = 0.325
c_b_eq = 0.369
c_a0 = 0.291
c_b0 = 0.394
d_a = 5.0
d_b = 100.0
t_end = 3e5

average_comp = ( (l_a*c_a0)+(l_b*c_b0) ) / (l_a + l_b)
print(f"average_comp: {average_comp}")
final_a_fraction = (average_comp - c_a_eq) / (c_b_eq - c_a_eq)
final_b_fraction = (c_b_eq - average_comp) / (c_b_eq - c_a_eq)
print(f"final_a_fraction: {final_a_fraction}")
print(f"final_b_fraction: {final_b_fraction}")

final_l_a = final_a_fraction * L
final_l_b = final_b_fraction * L
print(f"final_l_a: {final_l_a}")
print(f"final_l_b: {final_l_b}")

record_input=100
vIt_input=5000
verbose_input = False if record_input == False else True
bulkUpdateScheme_input= ['legacy', 'flux_form'][0]
modelsToUse = ['post_corr'] #, 'pre_basic', 'post_corr', 'pre_corr']
valid_models = {'post_basic', 'pre_basic', 'post_corr', 'pre_corr'}
unknown_models = sorted(set(modelsToUse) - valid_models)
if unknown_models:
    raise ValueError(f"Unknown model names in modelsToUse: {unknown_models}")
use_post_basic = 'post_basic' in modelsToUse
use_pre_basic = 'pre_basic' in modelsToUse
use_post_corr = 'post_corr' in modelsToUse
use_pre_corr = 'pre_corr' in modelsToUse
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

from kawin.diffusion.DiffusionParameters import DiffusionConstraints
constraints = DiffusionConstraints()
constraints.movingBoundaryThreshold = 0.3

save_basePath = pathlib.Path(r"C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\kawin\examples\modelSaves")
save_basePath.mkdir(parents=True, exist_ok=True)
cache_metadata_path = save_basePath / 'cache_metadata.json'
MODEL_CONFIGURATION_LOOKUP = {
    'post_basic': {
        'cache_label': 'post_model_basic',
        'fluxGradientMode': 'post_diffusion',
        'interfaceUpdate': 'basic',
        'plot_label': 'Post-Diffusion Basic',
        'color': 'tab:blue',
        'marker': 'o',
    },
    'pre_basic': {
        'cache_label': 'pre_model_basic',
        'fluxGradientMode': 'pre_diffusion',
        'interfaceUpdate': 'basic',
        'plot_label': 'Pre-Diffusion Basic',
        'color': 'tab:orange',
        'marker': 'x',
    },
    'post_corr': {
        'cache_label': 'post_model_corr',
        'fluxGradientMode': 'post_diffusion',
        'interfaceUpdate': 'lee_oh_corrected',
        'plot_label': 'Post-Diffusion Corrected',
        'color': 'tab:green',
        'marker': 'o',
    },
    'pre_corr': {
        'cache_label': 'pre_model_corr',
        'fluxGradientMode': 'pre_diffusion',
        'interfaceUpdate': 'lee_oh_corrected',
        'plot_label': 'Pre-Diffusion Corrected',
        'color': 'tab:red',
        'marker': 'x',
    },
}
common_cache_params = {
    'cache_version': MODEL_CACHE_VERSION,
    'model': 'MovingBoundaryFD1DModel',
    'elements': ['FE', 'CR'],
    'phases': ['ALPHA', 'BETA'],
    'temperature': 1000,
    'pstar': 0.5,
    'record': record_input,
    'iterator': 'explicitEulerIterator',
    't_end': t_end,
    'L': L,
    'N': N,
    'interface_position': interface_position,
    'c_a0': c_a0,
    'c_b0': c_b0,
    'c_a_eq': c_a_eq,
    'c_b_eq': c_b_eq,
    'd_a': d_a,
    'd_b': d_b,
    'constraints': diffusion_constraints_to_cache_dict(constraints),
}
post_model_basic = None
pre_model_basic = None
post_model_corr = None
pre_model_corr = None


def build_case_constraints(constraint_params):
    '''
    Builds a DiffusionConstraints object from serialized cache metadata.
    '''
    case_constraints = DiffusionConstraints()
    for key, value in constraint_params.items():
        setattr(case_constraints, key, value)
    return case_constraints


def build_case_config(case_overrides=None):
    '''
    Creates a simulation configuration for a parameter sweep case.
    '''
    case_overrides = {} if case_overrides is None else dict(case_overrides)
    config = {
        'temperature': 1000,
        'L': L,
        'N': N,
        'interface_position': interface_position,
        'c_a0': c_a0,
        'c_b0': c_b0,
        'c_a_eq': c_a_eq,
        'c_b_eq': c_b_eq,
        'd_a': d_a,
        'd_b': d_b,
        't_end': t_end,
        'record': record_input,
        'bulkUpdateScheme': bulkUpdateScheme_input,
        'pstar': 0.5,
        'modelsToUse': list(modelsToUse),
        'constraints': diffusion_constraints_to_cache_dict(constraints),
    }
    explicit_label = case_overrides.pop('label', None)
    constraint_override_keys = set(diffusion_constraints_to_cache_dict(constraints).keys())
    if 'constraints' in case_overrides:
        config['constraints'] = {
            **config['constraints'],
            **case_overrides.pop('constraints'),
        }
    for key in list(case_overrides.keys()):
        if key in constraint_override_keys:
            config['constraints'][key] = case_overrides.pop(key)
    config.update(case_overrides)
    if explicit_label is not None:
        config['label'] = explicit_label
    else:
        label_items = []
        for key, value in case_overrides.items():
            if key in {'modelsToUse', 'color'}:
                continue
            label_items.append(f"{key}={value}")
        for key, value in config['constraints'].items():
            if key in constraint_override_keys and value != diffusion_constraints_to_cache_dict(constraints)[key]:
                label_items.append(f"{key}={value}")
        config['label'] = ', '.join(label_items) if label_items else 'base case'
    return config


def run_cached_case(case_overrides=None):
    '''
    Runs or loads a set of model variants for one parameter-sweep case.
    '''
    config = build_case_config(case_overrides)
    case_constraints = build_case_constraints(config['constraints'])
    case_profile = ProfileBuilder([(StepProfile1D(config['interface_position'], config['c_a0'], config['c_b0']), 'CR')])
    case_mesh = CartesianFD1D(['CR'], [0, config['L']], config['N'] + 1)
    case_mesh.setResponseProfile(case_profile)
    case_therm = ConstantBinaryThermodynamics(
        ['ALPHA', 'BETA'],
        {'ALPHA': config['d_a'], 'BETA': config['d_b']},
        (config['c_a_eq'], config['c_b_eq']),
    )
    case_common_cache_params = {
        'cache_version': MODEL_CACHE_VERSION,
        'model': 'MovingBoundaryFD1DModel',
        'elements': ['FE', 'CR'],
        'phases': ['ALPHA', 'BETA'],
        'temperature': config['temperature'],
        'pstar': config['pstar'],
        'record': config['record'],
        'iterator': 'explicitEulerIterator',
        't_end': config['t_end'],
        'L': config['L'],
        'N': config['N'],
        'interface_position': config['interface_position'],
        'c_a0': config['c_a0'],
        'c_b0': config['c_b0'],
        'c_a_eq': config['c_a_eq'],
        'c_b_eq': config['c_b_eq'],
        'd_a': config['d_a'],
        'd_b': config['d_b'],
        'constraints': diffusion_constraints_to_cache_dict(case_constraints),
    }

    results = {}
    for model_name in config['modelsToUse']:
        model_config = MODEL_CONFIGURATION_LOOKUP[model_name]
        case_model = MovingBoundaryFD1DModel(
            case_mesh,
            ['FE', 'CR'],
            ['ALPHA', 'BETA'],
            case_therm,
            temperature=config['temperature'],
            interfacePosition=config['interface_position'],
            bulkUpdateScheme=config['bulkUpdateScheme'],
            fluxGradientMode=model_config['fluxGradientMode'],
            interfaceUpdate=model_config['interfaceUpdate'],
            pstar=config['pstar'],
            constraints=case_constraints,
            record=config['record'],
        )
        case_cache_params = {
            **case_common_cache_params,
            'cache_label': model_config['cache_label'],
            'bulkUpdateScheme': config['bulkUpdateScheme'],
            'fluxGradientMode': model_config['fluxGradientMode'],
            'interfaceUpdate': model_config['interfaceUpdate'],
        }
        case_save_path = save_basePath / register_cache_metadata(
            cache_metadata_path,
            model_config['cache_label'],
            case_cache_params,
        )
        if case_save_path.exists():
            case_model.load(case_save_path)
            print(f"Loaded {model_config['cache_label']} from cache: {case_save_path.name}")
        else:
            case_model.solve(config['t_end'], iterator=explicitEulerIterator, vIt=vIt_input, verbose=verbose_input)
            case_model.save(case_save_path)
            print(f"Saved {model_config['cache_label']} to cache: {case_save_path.name}")

        time_arr = np.array(case_model.interfaceData._time[:case_model.interfaceData.N+1], dtype=np.float64)
        position_arr = np.array(case_model.interfaceData._y[:case_model.interfaceData.N+1], dtype=np.float64)
        results[model_name] = {
            'model': case_model,
            't': time_arr,
            's': position_arr,
            'normalized_thickness': (config['L'] - position_arr) / (config['L'] - position_arr[0]),
        }

    return {
        'label': config['label'],
        'config': config,
        'results': results,
    }

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



if use_post_basic:
    post_model_basic = MovingBoundaryFD1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        therm,
        temperature=1000,
        interfacePosition=interface_position,
        bulkUpdateScheme=bulkUpdateScheme_input,
        fluxGradientMode='post_diffusion',
        interfaceUpdate='basic',
        pstar=0.5,
        constraints=constraints,
        record=record_input,
    )
    post_model_basic_cache_params = {
        **common_cache_params,
        'cache_label': 'post_model_basic',
        'bulkUpdateScheme': bulkUpdateScheme_input,
        'fluxGradientMode': 'post_diffusion',
        'interfaceUpdate': 'basic',
    }
    post_model_basic_save_path = save_basePath / register_cache_metadata(cache_metadata_path, 'post_model_basic', post_model_basic_cache_params)
    t0_post_basic_solve=time.perf_counter()
    if post_model_basic_save_path.exists():
        post_model_basic.load(post_model_basic_save_path)
        print(f"Loaded post_model_basic from cache: {post_model_basic_save_path.name}")
    else:
        post_model_basic.solve(t_end, iterator=explicitEulerIterator, vIt=vIt_input, verbose=verbose_input)
        post_model_basic.save(post_model_basic_save_path)
        print(f"Saved post_model_basic to cache: {post_model_basic_save_path.name}")
    t1_post_basic_solve=time.perf_counter()
    print(f"Post-Diffusion Basic Solve Time: {t1_post_basic_solve - t0_post_basic_solve}")
if use_pre_basic:
    pre_model_basic = MovingBoundaryFD1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        therm,
        temperature=1000,
        interfacePosition=interface_position,
        bulkUpdateScheme=bulkUpdateScheme_input,
        fluxGradientMode='pre_diffusion',
        interfaceUpdate='basic',
        pstar=0.5,
        constraints=constraints,
        record=record_input,
    )
    pre_model_basic_cache_params = {
        **common_cache_params,
        'cache_label': 'pre_model_basic',
        'bulkUpdateScheme': bulkUpdateScheme_input,
        'fluxGradientMode': 'pre_diffusion',
        'interfaceUpdate': 'basic',
    }
    pre_model_basic_save_path = save_basePath / register_cache_metadata(cache_metadata_path, 'pre_model_basic', pre_model_basic_cache_params)
    if pre_model_basic_save_path.exists():
        pre_model_basic.load(pre_model_basic_save_path)
        print(f"Loaded pre_model_basic from cache: {pre_model_basic_save_path.name}")
    else:
        pre_model_basic.solve(t_end, iterator=explicitEulerIterator, vIt=vIt_input, verbose=verbose_input)
        pre_model_basic.save(pre_model_basic_save_path)
        print(f"Saved pre_model_basic to cache: {pre_model_basic_save_path.name}")

if use_post_corr:
    post_model_corr = MovingBoundaryFD1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        therm,
        temperature=1000,
        interfacePosition=interface_position,
        bulkUpdateScheme=bulkUpdateScheme_input,
        fluxGradientMode='post_diffusion',
        interfaceUpdate='lee_oh_corrected',
        pstar=0.5,
        constraints=constraints,
        record=record_input,
    )
    post_model_corr_cache_params = {
        **common_cache_params,
        'cache_label': 'post_model_corr',
        'bulkUpdateScheme': bulkUpdateScheme_input,
        'fluxGradientMode': 'post_diffusion',
        'interfaceUpdate': 'lee_oh_corrected',
    }
    post_model_corr_save_path = save_basePath / register_cache_metadata(cache_metadata_path, 'post_model_corr', post_model_corr_cache_params)
    t0_post_corr_solve=time.perf_counter()
    if post_model_corr_save_path.exists():
        post_model_corr.load(post_model_corr_save_path)
        print(f"Loaded post_model_corr from cache: {post_model_corr_save_path.name}")
    else:
        post_model_corr.solve(t_end, iterator=explicitEulerIterator, vIt=vIt_input, verbose=verbose_input)
        post_model_corr.save(post_model_corr_save_path)
        print(f"Saved post_model_corr to cache: {post_model_corr_save_path.name}")
    t1_post_corr_solve=time.perf_counter()
    print(f"Post-Diffusion Corrected Solve Time: {t1_post_corr_solve - t0_post_corr_solve}")
if use_pre_corr:
    pre_model_corr = MovingBoundaryFD1DModel(
        mesh,
        ['FE', 'CR'],
        ['ALPHA', 'BETA'],
        therm,
        temperature=1000,
        interfacePosition=interface_position,
        bulkUpdateScheme=bulkUpdateScheme_input,
        fluxGradientMode='pre_diffusion',
        interfaceUpdate='lee_oh_corrected',
        pstar=0.5,
        constraints=constraints,
        record=record_input,
    )
    pre_model_corr_cache_params = {
        **common_cache_params,
        'cache_label': 'pre_model_corr',
        'bulkUpdateScheme': bulkUpdateScheme_input,
        'fluxGradientMode': 'pre_diffusion',
        'interfaceUpdate': 'lee_oh_corrected',
    }
    pre_model_corr_save_path = save_basePath / register_cache_metadata(cache_metadata_path, 'pre_model_corr', pre_model_corr_cache_params)
    if pre_model_corr_save_path.exists():
        pre_model_corr.load(pre_model_corr_save_path)
        print(f"Loaded pre_model_corr from cache: {pre_model_corr_save_path.name}")
    else:
        pre_model_corr.solve(t_end, iterator=explicitEulerIterator, vIt=vIt_input, verbose=verbose_input)
        pre_model_corr.save(pre_model_corr_save_path)
        print(f"Saved pre_model_corr to cache: {pre_model_corr_save_path.name}")
t1_overall=time.perf_counter()
print(f"Overall Time: {t1_overall - t0_overall}")

t_post_basic = s_post_basic = sqrt_t_post_basic = delta_s_post_basic = None
t_pre_basic = s_pre_basic = sqrt_t_pre_basic = delta_s_pre_basic = None
t_post_corr = s_post_corr = sqrt_t_post_corr = delta_s_post_corr = None
t_pre_corr = s_pre_corr = sqrt_t_pre_corr = delta_s_pre_corr = None

if use_post_basic:
    t_post_basic = np.array(post_model_basic.interfaceData._time[:post_model_basic.interfaceData.N+1], dtype=np.float64)
    s_post_basic = np.array(post_model_basic.interfaceData._y[:post_model_basic.interfaceData.N+1], dtype=np.float64)
    sqrt_t_post_basic = np.sqrt(t_post_basic)
    delta_s_post_basic = s_post_basic - s_post_basic[0]

if use_pre_basic:
    t_pre_basic = np.array(pre_model_basic.interfaceData._time[:pre_model_basic.interfaceData.N+1], dtype=np.float64)
    s_pre_basic = np.array(pre_model_basic.interfaceData._y[:pre_model_basic.interfaceData.N+1], dtype=np.float64)
    sqrt_t_pre_basic = np.sqrt(t_pre_basic)
    delta_s_pre_basic = s_pre_basic - s_pre_basic[0]

if use_post_corr:
    t_post_corr = np.array(post_model_corr.interfaceData._time[:post_model_corr.interfaceData.N+1], dtype=np.float64)
    s_post_corr = np.array(post_model_corr.interfaceData._y[:post_model_corr.interfaceData.N+1], dtype=np.float64)
    sqrt_t_post_corr = np.sqrt(t_post_corr)
    delta_s_post_corr = s_post_corr - s_post_corr[0]

if use_pre_corr:
    t_pre_corr = np.array(pre_model_corr.interfaceData._time[:pre_model_corr.interfaceData.N+1], dtype=np.float64)
    s_pre_corr = np.array(pre_model_corr.interfaceData._y[:pre_model_corr.interfaceData.N+1], dtype=np.float64)
    sqrt_t_pre_corr = np.sqrt(t_pre_corr)
    delta_s_pre_corr = s_pre_corr - s_pre_corr[0]

# for modelName in modelsToUse:
#     if modelName=="post_basic":
#         np.savez(rf"C:\Users\samth\Downloads\{modelName}_1e3_{bulkUpdateScheme_input}.npz", x_axis=t_post_basic, y_axis=s_post_basic)
#     elif modelName=="pre_basic":
#         np.savez(rf"C:\Users\samth\Downloads\{modelName}_1e3_{bulkUpdateScheme_input}.npz", x_axis=t_pre_basic, y_axis=s_pre_basic)
#     elif modelName=="post_corr":
#         np.savez(rf"C:\Users\samth\Downloads\{modelName}_1e3_{bulkUpdateScheme_input}.npz", x_axis=t_post_corr, y_axis=s_post_corr)
#     elif modelName=="pre_corr":
#         np.savez(rf"C:\Users\samth\Downloads\{modelName}_1e3_{bulkUpdateScheme_input}.npz", x_axis=t_pre_corr, y_axis=s_pre_corr)
#     else:
#         raise

#%%
beta = solve_beta(c_a0, c_b0, c_a_eq, c_b_eq, d_a, d_b)
analytic_delta_s = lambda sqrt_t: 2.0 * beta * sqrt_t

selected_models = []
if use_post_basic:
    selected_models.append(('post_basic', post_model_basic, t_post_basic, s_post_basic, sqrt_t_post_basic, delta_s_post_basic))
if use_pre_basic:
    selected_models.append(('pre_basic', pre_model_basic, t_pre_basic, s_pre_basic, sqrt_t_pre_basic, delta_s_pre_basic))
if use_post_corr:
    selected_models.append(('post_corr', post_model_corr, t_post_corr, s_post_corr, sqrt_t_post_corr, delta_s_post_corr))
if use_pre_corr:
    selected_models.append(('pre_corr', pre_model_corr, t_pre_corr, s_pre_corr, sqrt_t_pre_corr, delta_s_pre_corr))

if not selected_models:
    raise ValueError("modelsToUse must include at least one model.")

for _, model_obj, _, _, _, _ in selected_models:
    assert model_obj.mesh.N-1 == N

num_points_lookup = {}
sqrt_t_sub_lookup = {}
delta_s_sub_lookup = {}
analytic_sub_lookup = {}
for model_name, model_obj, t_arr, _, sqrt_t_arr, delta_s_arr in selected_models:
    num_points = max(5, min(len(sqrt_t_arr), math.floor(min(interface_position, L - interface_position) / (L / N)) - 5))
    num_points_lookup[model_name] = num_points
    sqrt_t_sub_lookup[model_name] = sqrt_t_arr[:num_points]
    delta_s_sub_lookup[model_name] = delta_s_arr[:num_points]
    analytic_sub_lookup[model_name] = analytic_delta_s(sqrt_t_sub_lookup[model_name])[:num_points]

reference_model_name = selected_models[0][0]

SEMIINFINITE_ASSUMPTION_VIOLATED=None
if use_post_corr and (np.ravel(post_model_corr.data.y(t_post_corr[num_points_lookup['post_corr']-1]))[[0,-1]]==np.array([c_a0, c_b0])).all()!=True:
    np.ravel(post_model_corr.data.y(t_post_corr[num_points_lookup['post_corr']-1]))
    SEMIINFINITE_ASSUMPTION_VIOLATED=True
    print("WARNING: SEMIINFINITE ASSUMPTION MAY BE VIOLATED")

#%%
fig, ax = plt.subplots()
if use_post_basic:
    ax.plot(sqrt_t_sub_lookup['post_basic'], delta_s_sub_lookup['post_basic'], 'o', label='MovingBoundaryFD1D Post-Diffusion Basic', color='tab:blue')
if use_post_corr:
    ax.plot(sqrt_t_sub_lookup['post_corr'], delta_s_sub_lookup['post_corr'], 'o', label='MovingBoundaryFD1D Post-Diffusion Corrected', color='tab:green')
if use_pre_basic:
    ax.plot(sqrt_t_sub_lookup['pre_basic'], delta_s_sub_lookup['pre_basic'], 'x', label='MovingBoundaryFD1D Pre-Diffusion Basic', color='tab:orange')
if use_pre_corr:
    ax.plot(sqrt_t_sub_lookup['pre_corr'], delta_s_sub_lookup['pre_corr'], 'x', label='MovingBoundaryFD1D Pre-Diffusion Corrected', color='tab:red')
ax.plot(sqrt_t_sub_lookup[reference_model_name], analytic_sub_lookup[reference_model_name], label='Analytic', color='k')
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
if SEMIINFINITE_ASSUMPTION_VIOLATED:
    ax.text(0.02, 0.3, "SEMIINFINITE ASSUMPTION VIOLATED", transform=ax.transAxes, color='red')
ax.set_xlabel('sqrt(t)')
ax.set_ylabel('Change in interface position')
# ax.set_ylim(-0.007, 0.0015)
# ax.set_xlim(0, 0.012)
ax.legend()
plt.tight_layout()
plt.show()

#%%
idealized_mass_funcOfStartingInterfacePosition = lambda startingInterfacePosition: (c_a0 * startingInterfacePosition) + (c_b0 * (L - startingInterfacePosition))
idealized_mass = idealized_mass_funcOfStartingInterfacePosition(interface_position)

fig2, ax2 = plt.subplots()
pos_and_mass_error_lookup = {}
for model_name, model_obj, t_arr, s_arr, _, _ in selected_models:
    num_points = num_points_lookup[model_name]
    pos_and_mass_error_lookup[model_name] = np.array(
        [
            (
                ((s_arr[i] - s_arr[0]) - 2 * beta * np.sqrt(t)) / (2 * beta * np.sqrt(t)),
                (model_obj.getTotalMass(time=t) - model_obj._initialInventory) / model_obj._initialInventory,
            )
            for i, t in enumerate(t_arr)
        ]
    )[:num_points]

if use_post_basic:
    ax2.plot(pos_and_mass_error_lookup['post_basic'][:,1], pos_and_mass_error_lookup['post_basic'][:,0], 'o-', fillstyle='none', label='Post-Diffusion Basic', color='tab:blue')
if use_post_corr:
    ax2.plot(pos_and_mass_error_lookup['post_corr'][:,1], pos_and_mass_error_lookup['post_corr'][:,0], 'o-', fillstyle='none', label='Post-Diffusion Corrected', color='tab:green')
if use_pre_basic:
    ax2.plot(pos_and_mass_error_lookup['pre_basic'][:,1], pos_and_mass_error_lookup['pre_basic'][:,0], 'x-', fillstyle='none', label='Pre-Diffusion Basic', color='tab:orange')
if use_pre_corr:
    ax2.plot(pos_and_mass_error_lookup['pre_corr'][:,1], pos_and_mass_error_lookup['pre_corr'][:,0], 'x-', fillstyle='none', label='Pre-Diffusion Corrected', color='tab:red')
for model_name, color in [('post_basic', 'tab:blue'), ('post_corr', 'tab:green'), ('pre_basic', 'tab:orange'), ('pre_corr', 'tab:red')]:
    if model_name in pos_and_mass_error_lookup and len(pos_and_mass_error_lookup[model_name]) > 1:
        ax2.scatter(pos_and_mass_error_lookup[model_name][:,1][1], pos_and_mass_error_lookup[model_name][:,0][1], marker='s', color=color)
ax2.hlines(0, ax2.get_xlim()[0], ax2.get_xlim()[1], colors='gray', linewidth=1, alpha=0.4, zorder=-1)
ax2.vlines(0, ax2.get_ylim()[0], ax2.get_ylim()[1], colors='gray', linewidth=1, alpha=0.4, zorder=-1)
ax2.set_xlabel('Mass Error (relative to initial inventory)', fontsize=12)
ax2.set_ylabel('Interface Displacement Error (relative to analytic solution)', fontsize=12)
ax2.legend()
plt.tight_layout()
for model_name in ['post_basic', 'pre_basic', 'post_corr', 'pre_corr']:
    if model_name in pos_and_mass_error_lookup:
        print(f"{model_name} final errors: {pos_and_mass_error_lookup[model_name][-1]}")

for model_name, model_obj, _, _, _, _ in selected_models:
    print((f"{model_name} inventory", model_obj._initialInventory, model_obj.getTotalMass(), model_obj.getTotalMass()-model_obj._initialInventory))

#%%
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
        bulkUpdateScheme=bulkUpdateScheme_input,
        fluxGradientMode='post_diffusion',
        interfaceUpdate='basic',
        pstar=0.5,
        record=True,
    )
    return model_for_initMass._initialInventory

reference_inventory_model = selected_models[0][1]
initial_mass_funcOfStartingInterfacePosition(interface_position)-reference_inventory_model._initialInventory

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
parameterSweepConfigs = [
    {'N': 90, 'movingBoundaryThreshold': 0.3},
    {'N': 100, 'movingBoundaryThreshold': 0.3},
    {'N': 150, 'movingBoundaryThreshold': 0.3},
    {'N': 100, 'movingBoundaryThreshold': 0.25},
    {'N': 100, 'movingBoundaryThreshold': 0.4},
    {'N': 150, 'movingBoundaryThreshold': 0.4},
    {'N': 175, 'movingBoundaryThreshold': 0.4},
    {'N': 200, 'movingBoundaryThreshold': 0.4},
    {'N': 250, 'movingBoundaryThreshold': 0.4},
]

fig4, ax4 = plt.subplots(figsize=(12, 10))
downSample_fig8Rep=True
markersize=3
if downSample_fig8Rep==True:
    log_indices = lambda arr: np.unique(np.logspace(0, np.log10(len(arr)-1), num=2000).astype(int))
else:
    log_indices = lambda arr: np.arange(len(arr))
if use_post_basic:
    ax4.plot(t_post_basic[log_indices(t_post_basic)], (L-s_post_basic[log_indices(t_post_basic)])/(L-s_post_basic[0]), 'o-', markersize=markersize, fillstyle='none', label='Post-Diffusion Basic', color='tab:blue')
if use_pre_basic:
    ax4.plot(t_pre_basic[log_indices(t_pre_basic)], (L-s_pre_basic[log_indices(t_pre_basic)])/(L-s_pre_basic[0]), 'x-', markersize=markersize, fillstyle='none', label='Pre-Diffusion Basic', color='tab:orange')
if use_post_corr:
    ax4.plot(t_post_corr[log_indices(t_post_corr)], (L-s_post_corr[log_indices(t_post_corr)])/(L-s_post_corr[0]), 'o-', markersize=markersize, fillstyle='none', label='Post-Diffusion Corrected', color='tab:green')
if use_pre_corr:
    ax4.plot(t_pre_corr[log_indices(t_pre_corr)], (L-s_pre_corr[log_indices(t_pre_corr)])/(L-s_pre_corr[0]), 'x-', markersize=markersize, fillstyle='none', label='Pre-Diffusion Corrected', color='tab:red')

fig8_presentMethod_df.plot(x='t', y='normalized_thickness', ax=ax4, label='Lee and Oh 1996 Fig. 8 Present Method Curve', color='k')
figure8_presentMethod_oldExtract_df = pd.read_csv(r"C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\finiteDifference\LeeAndOh_figure8_presentMethod.csv", low_memory=False, names=['t', 'normalized_thickness'])
figure8_presentMethod_oldExtract_df = figure8_presentMethod_oldExtract_df.sort_values(by=['t'])
figure8_presentMethod_oldExtract_df.plot(x='t', y='normalized_thickness', ax=ax4, label='Lee and Oh 1996 Fig. 8 Present Method Curve (OLD EXTRACTION)', color='dimgray', zorder=-1)

ax4.legend()
ax4.set_xscale('log')
ax4.set_xlim(5, 1e6)
ax4.set_ylim(0, 1.4)
fig4.savefig(rf"C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\kawin\examples\replicationOfFig8_{'downSampled' if downSample_fig8Rep==True else 'full'}.svg")

parameter_sweep_results = [run_cached_case(case_config) for case_config in parameterSweepConfigs]
fig4_sweep, ax4_sweep = plt.subplots(figsize=(12, 10))
line_styles = ['-', '--', ':', '-.']
parameter_set_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink']
final_thickness_summary_lines = []
for case_index, case_result in enumerate(parameter_sweep_results):
    case_config = case_result['config']
    case_label = case_result['label']
    line_style = line_styles[case_index % len(line_styles)]
    case_color = parameter_set_colors[case_index % len(parameter_set_colors)]
    for model_name, series in case_result['results'].items():
        model_plot_config = MODEL_CONFIGURATION_LOOKUP[model_name]
        sample_indices = log_indices(series['t'])
        ax4_sweep.plot(
            series['t'][sample_indices],
            series['normalized_thickness'][sample_indices],
            linestyle=line_style,
            marker=model_plot_config['marker'],
            markersize=markersize,
            fillstyle='none',
            color=case_color,
            label=f"{model_plot_config['plot_label']} ({case_label})",
            alpha=0.5
        )
        final_thickness_summary_lines.append(
            f"{case_label} | {model_plot_config['plot_label']}: {series['normalized_thickness'][-1]:.6f}"
        )

fig8_presentMethod_df.plot(
    x='t',
    y='normalized_thickness',
    ax=ax4_sweep,
    label='Lee and Oh 1996 Fig. 8 Present Method Curve',
    color='k',
    linewidth=1.5,
)
figure8_presentMethod_oldExtract_df.plot(
    x='t',
    y='normalized_thickness',
    ax=ax4_sweep,
    label='Lee and Oh 1996 Fig. 8 Present Method Curve (OLD EXTRACTION)',
    color='dimgray',
    zorder=-1,
)
parameter_sweep_text = '\n'.join([case_result['label'] for case_result in parameter_sweep_results])
ax4_sweep.text(
    0.02,
    0.98,
    f"Parameter sweep\n{parameter_sweep_text}",
    transform=ax4_sweep.transAxes,
    va='top',
    ha='left',
    fontsize=9,
    bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.85, 'edgecolor': '0.7'},
)
ax4_sweep.text(
    0.02,
    0.72,
    "Final normalized thickness\n" + '\n'.join(final_thickness_summary_lines),
    transform=ax4_sweep.transAxes,
    va='top',
    ha='left',
    fontsize=8,
    bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.85, 'edgecolor': '0.7'},
)
ax4_sweep.legend()
ax4_sweep.set_xscale('log')
ax4_sweep.set_xlim(5, 1e6)
ax4_sweep.set_ylim(0, 1.4)
ax4_sweep.set_xlabel('Time')
ax4_sweep.set_ylabel('Normalized thickness')
fig4_sweep.tight_layout()
fig4_sweep.savefig(
    rf"C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\kawin\examples\replicationOfFig8_parameterSweep_{'downSampled' if downSample_fig8Rep==True else 'full'}.svg"
)

if use_post_basic:
    (L-s_post_basic[-1])/(L-s_post_basic[0])
if use_post_corr:
    (L-s_post_corr[-1])/(L-s_post_corr[0])
if use_pre_corr:
    (L-s_pre_corr[-1])/(L-s_pre_corr[0])
fig8_presentMethod_df['normalized_thickness'].iloc[-1]
figure8_presentMethod_oldExtract_df['normalized_thickness'].iloc[-1]

#%%
''' Quantitative Comparison of difference in t-s curves '''
lstOf_indx_TS_arrs_dict = []
if use_post_basic:
    lstOf_indx_TS_arrs_dict.append({'t':t_post_basic.copy(), 's':s_post_basic.copy(), 'name':'post_basic'})
if use_pre_basic:
    lstOf_indx_TS_arrs_dict.append({'t':t_pre_basic.copy(), 's':s_pre_basic.copy(), 'name':'pre_basic'})
if use_post_corr:
    lstOf_indx_TS_arrs_dict.append({'t':t_post_corr.copy(), 's':s_post_corr.copy(), 'name':'post_corr'})
if use_pre_corr:
    lstOf_indx_TS_arrs_dict.append({'t':t_pre_corr.copy(), 's':s_pre_corr.copy(), 'name':'pre_corr'})
lstOf_indx_TS_arrs_dict.extend([
    {'t':fig8_presentMethod_df['t'].to_numpy().copy(), 's_norm':fig8_presentMethod_df['normalized_thickness'].to_numpy().copy(), 'name':'fig8_currentExtract'},
    {'t':figure8_presentMethod_oldExtract_df['t'].to_numpy().copy(), 's_norm':figure8_presentMethod_oldExtract_df['normalized_thickness'].to_numpy().copy(), 'name':'fig8_oldExtract'},
])
TS_df = pd.DataFrame(lstOf_indx_TS_arrs_dict)
TS_df.loc[TS_df['s_norm'].isna(), 's_norm'] = TS_df.loc[TS_df['s_norm'].isna(), 's'].apply(lambda s_arr: (L-s_arr)/(L-s_arr[0]))
from scipy.interpolate import splrep, splev
def buildAndEvalSpline(x, y, xToEval):
    spl = splrep(x, y, k=3)
    return splev(xToEval, spl, ext=2)

commonLowerBound_t = TS_df['t'].apply(min).max()
commonUpperBound_t = TS_df['t'].apply(max).min()
T_toEvalAt_1 = np.logspace(-1, -2, 1000)-1e-10
T_toEvalAt_common = np.logspace(np.log10(commonLowerBound_t) + 1e-6, np.log10(commonUpperBound_t) - 1e-6, 1000)
TS_df['sNorm_splEval'] = TS_df.apply(lambda r: buildAndEvalSpline(x=r['t'], y=r['s_norm'], xToEval=T_toEvalAt_common), axis=1)


MSE_func = lambda arr1, arr2: np.linalg.norm(arr1-arr2)/len(arr1)

# currentExtract_splEval = buildAndEvalSpline(x=fig8_presentMethod_df['t'].to_numpy(), y=fig8_presentMethod_df['normalized_thickness'].to_numpy(), xToEval=np.logspace(np.log10(fig8_presentMethod_df['t'].min())+1e-6, np.log10(figure8_presentMethod_oldExtract_df['t'].max())-1e-6, 1000))
# oldExtract_splEval = buildAndEvalSpline(x=figure8_presentMethod_oldExtract_df['t'].to_numpy(), y=figure8_presentMethod_oldExtract_df['y'].to_numpy(), xToEval=np.logspace(np.log10(fig8_presentMethod_df['t'].min())+1e-6, np.log10(figure8_presentMethod_oldExtract_df['t'].max())-1e-6, 1000))
# print(f"MSE_func(currentExtract_splEval, oldExtract_splEval): {MSE_func(currentExtract_splEval, oldExtract_splEval)}")

MSE_dict={}
for indx1 in range(len(TS_df)):
    for indx2 in range(len(TS_df)):
        MSE_val = MSE_func(TS_df['sNorm_splEval'].iloc[indx1], TS_df['sNorm_splEval'].iloc[indx2])
        MSE_dict.update({(TS_df['name'].iloc[indx1], TS_df['name'].iloc[indx2]):MSE_val})
        
# MSE_df = pd.DataFrame(columns=range(len(TS_df)), index=range(len(TS_df)))
MSE_df = pd.DataFrame(columns=TS_df['name'].to_list(), index=TS_df['name'].to_list())
for MSE_key, MSE_val in MSE_dict.items():
    MSE_df.loc[MSE_key[0], MSE_key[1]] = MSE_val
MSE_df=MSE_df.astype(float)
# display(MSE_df/MSE_df.loc['fig8_currentExtract', 'fig8_oldExtract'])
lowerTriangular_mask = np.tril(np.ones(MSE_df.shape)).astype(bool)
display((MSE_df/MSE_df.loc['fig8_currentExtract', 'fig8_oldExtract']).where(~lowerTriangular_mask))

assert np.all(np.isfinite(sqrt_t_sub_lookup[reference_model_name]))
assert np.all(np.isfinite(delta_s_sub_lookup[reference_model_name]))
assert np.all(np.isfinite(analytic_sub_lookup[reference_model_name]))
# plt.close(fig)


# %%
