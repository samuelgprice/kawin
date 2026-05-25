#%%

# %matplotlib inline
#
# If you run this file in IPython/Jupyter, you can uncomment the next line.
# %config InlineBackend.figure_format = 'svg'
import pathlib
import time
import json
import hashlib
import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from kawin.diffusion import MovingBoundaryFD1DModel
from kawin.diffusion.DiffusionParameters import DiffusionConstraints, TemperatureParameters
from kawin.diffusion.Plot import plot1D
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.diffusion.mesh.MovingBoundaryFD1D import get_moving_boundary_fd_geometry
from kawin.solver import explicitEulerIterator
from kawin.thermo import MulticomponentThermodynamics
from pycalphad import Database, ternplot
from pycalphad import variables as v

def debugInPlace():
    try:
        import debugpy
        # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        debugpy.breakpoint()
        print('break on this line')
    except:
        pass

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path.cwd()
# EXAMPLES_DIR = pathlib.Path(r"c:/Users/samth/OneDrive - Northwestern University/WS_DL/Lab Data/Price/code/kawin/examples")

pstar_input=0.01

class ApproximateTernaryTieLineThermodynamics:
    """
    Lightweight ternary thermodynamics surrogate for moving-boundary FDM tests.

    This class mimics the small public interface used by
    ``MovingBoundaryFD1DModel`` for ternary substitutions:

    - ``getInterdiffusivity`` supports either constant per-phase matrices
      (single tie-line mode) or KNN-backed nearest-sampled per-phase matrices.
    - ``getInterfacialComposition`` returns a tie-line that moves smoothly
      along a user-defined family parameterized by the interface probe
      composition.

    The intent is to make the ternary moving-boundary algorithm easy to test
    and debug without paying for repeated thermodynamic and mobility solves.
    """

    def __init__(
        self,
        phases,
        left_probe,
        right_probe,
        left_tieline_start,
        left_tieline_end,
        right_tieline_start,
        right_tieline_end,
        diffusivities,
        sampled_lambdas=None,
        sampled_left_tielines=None,
        sampled_right_tielines=None,
        phase_tieline_start=None,
        phase_tieline_end=None,
        sampled_phase_tielines=None,
        diffusivity_mode=None,
        sampled_diffusivity_compositions=None,
        sampled_diffusivities=None,
        sampled_diffusivity_tieline_lambdas=None,
        tdb_ForPlotting=None,
    ):
        self.phases = list(phases)
        if len(self.phases) != 2:
            raise ValueError("ApproximateTernaryTieLineThermodynamics expects exactly two phases.")
        self.matrix_phase = self.phases[0]
        self.precip_phase = self.phases[1]
        self.left_probe = np.asarray(left_probe, dtype=np.float64)
        self.right_probe = np.asarray(right_probe, dtype=np.float64)
        # Backward-compatible input aliases are still accepted, but internal
        # storage is phase-labeled to avoid "left/right tie-line" ambiguity.
        if phase_tieline_start is None:
            phase_tieline_start = {
                self.matrix_phase: np.asarray(left_tieline_start, dtype=np.float64),
                self.precip_phase: np.asarray(right_tieline_start, dtype=np.float64),
            }
        if phase_tieline_end is None:
            phase_tieline_end = {
                self.matrix_phase: np.asarray(left_tieline_end, dtype=np.float64),
                self.precip_phase: np.asarray(right_tieline_end, dtype=np.float64),
            }
        self.phase_tieline_start = {
            phase: np.asarray(value, dtype=np.float64) for phase, value in phase_tieline_start.items()
        }
        self.phase_tieline_end = {
            phase: np.asarray(value, dtype=np.float64) for phase, value in phase_tieline_end.items()
        }
        if self.matrix_phase not in self.phase_tieline_start or self.precip_phase not in self.phase_tieline_start:
            raise ValueError("phase_tieline_start must include matrix and precipitate phase keys.")
        if self.matrix_phase not in self.phase_tieline_end or self.precip_phase not in self.phase_tieline_end:
            raise ValueError("phase_tieline_end must include matrix and precipitate phase keys.")
        # Legacy aliases kept for plotting/printing paths elsewhere in this file.
        self.left_tieline_start = self.phase_tieline_start[self.matrix_phase]
        self.left_tieline_end = self.phase_tieline_end[self.matrix_phase]
        self.right_tieline_start = self.phase_tieline_start[self.precip_phase]
        self.right_tieline_end = self.phase_tieline_end[self.precip_phase]
        self.probe_direction = self.right_probe - self.left_probe
        self.diffusivities = {
            phase: np.asarray(matrix, dtype=np.float64) for phase, matrix in diffusivities.items()
        }
        self.diffusivity_mode = str(diffusivity_mode)
        if self.diffusivity_mode not in {"single_tieline", "nearest_sample"}:
            raise ValueError("diffusivity_mode must be either 'single_tieline' or 'nearest_sample'.")
        self.sampled_diffusivity_compositions_by_context = {}
        self.sampled_diffusivities_by_context = {}
        self._diffusivity_nn_models = {}
        if self.diffusivity_mode == "nearest_sample":
            if sampled_diffusivity_compositions is None or sampled_diffusivities is None:
                raise ValueError(
                    "Nearest-sample mode requires sampled_diffusivity_compositions and sampled_diffusivities."
                )
            if sampled_diffusivity_tieline_lambdas is None:
                raise ValueError(
                    "Nearest-sample mode requires sampled_diffusivity_tieline_lambdas so interface/general pools can be derived."
                )
            n_interface = int(len(np.asarray(sampled_diffusivity_tieline_lambdas, dtype=np.float64).reshape(-1)))
            if n_interface < 1:
                raise ValueError("Nearest-sample mode requires at least one diffusivity tie-line lambda.")
            sampled_diffusivity_compositions_general = {
                phase: np.asarray(values, dtype=np.float64)
                for phase, values in sampled_diffusivity_compositions.items()
            }
            sampled_diffusivities_general = {
                phase: np.asarray(values, dtype=np.float64)
                for phase, values in sampled_diffusivities.items()
            }
            sampled_diffusivity_compositions_interface = {}
            sampled_diffusivities_interface = {}
            for phase in self.phases:
                phase_x = sampled_diffusivity_compositions_general[phase]
                phase_d = sampled_diffusivities_general[phase]
                if n_interface > phase_x.shape[0]:
                    raise ValueError(
                        f"Not enough sampled diffusivity points for phase {phase}: need at least {n_interface}, got {phase_x.shape[0]}."
                    )
                sampled_diffusivity_compositions_interface[phase] = np.asarray(phase_x[:n_interface], dtype=np.float64)
                sampled_diffusivities_interface[phase] = np.asarray(phase_d[:n_interface], dtype=np.float64)

            self.sampled_diffusivity_compositions_by_context = {
                "interface": sampled_diffusivity_compositions_interface,
                "general": sampled_diffusivity_compositions_general,
            }
            self.sampled_diffusivities_by_context = {
                "interface": sampled_diffusivities_interface,
                "general": sampled_diffusivities_general,
            }
            for context in ["interface", "general"]:
                for phase in self.phases:
                    if phase not in self.sampled_diffusivity_compositions_by_context[context] or phase not in self.sampled_diffusivities_by_context[context]:
                        raise ValueError(f"Nearest-sample diffusivity mode requires sampled data for phase {phase} in context '{context}'.")
                    comps = self.sampled_diffusivity_compositions_by_context[context][phase]
                    mats = self.sampled_diffusivities_by_context[context][phase]
                    if comps.ndim != 2:
                        raise ValueError(f"sampled_diffusivity_compositions[{context}][{phase}] must be 2D.")
                    if mats.ndim != 3:
                        raise ValueError(f"sampled_diffusivities[{context}][{phase}] must be 3D.")
                    if comps.shape[0] != mats.shape[0]:
                        raise ValueError(f"Sample count mismatch for phase {phase} in context '{context}'.")
                    if comps.shape[0] < 1:
                        raise ValueError(f"No sampled diffusivity points were provided for phase {phase} in context '{context}'.")
        if self.diffusivity_mode == "nearest_sample":
            self._build_diffusivity_nn_models()
        self.sampled_lambdas = None if sampled_lambdas is None else np.asarray(sampled_lambdas, dtype=np.float64).reshape(-1)
        if sampled_phase_tielines is None:
            if sampled_left_tielines is not None and sampled_right_tielines is not None:
                sampled_phase_tielines = {
                    self.matrix_phase: np.asarray(sampled_left_tielines, dtype=np.float64),
                    self.precip_phase: np.asarray(sampled_right_tielines, dtype=np.float64),
                }
            else:
                sampled_phase_tielines = None
        self.sampled_phase_tielines = None
        if sampled_phase_tielines is not None:
            self.sampled_phase_tielines = {
                phase: np.asarray(values, dtype=np.float64) for phase, values in sampled_phase_tielines.items()
            }
        self.sampled_left_tielines = None if self.sampled_phase_tielines is None else self.sampled_phase_tielines[self.matrix_phase]
        self.sampled_right_tielines = None if self.sampled_phase_tielines is None else self.sampled_phase_tielines[self.precip_phase]
        self.tdb_forPlotting = tdb_ForPlotting
        if self.tdb_forPlotting is not None:
            self.db_forPlotting = Database(str(self.tdb_forPlotting))
        else:
            self.db_forPlotting = None

    def clearCache(self):
        """Matches the thermodynamics API; there is no cache to clear."""
        return

    def _build_diffusivity_nn_models(self):
        """
        Builds one KNN index per phase for fast nearest-sample diffusivity lookup.
        """
        self._diffusivity_nn_models = {"interface": {}, "general": {}}
        for context in ["interface", "general"]:
            for phase in self.phases:
                samples_x = np.asarray(self.sampled_diffusivity_compositions_by_context[context][phase], dtype=np.float64)
                if samples_x.ndim != 2 or samples_x.shape[0] < 1:
                    raise ValueError(f"sampled_diffusivity_compositions[{context}][{phase}] must be a non-empty 2D array.")
                model = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
                model.fit(samples_x)
                self._diffusivity_nn_models[context][phase] = model

    def _lambda_from_probe(self, x):
        """
        Maps the ternary interface probe composition onto the tie-line family.

        The moving-boundary ternary solver searches over a probe composition
        between the adjacent left and right bulk nodes. This helper projects the
        probe onto that line segment and clips the family parameter to ``[0, 1]``.
        """
        probe = np.asarray(x, dtype=np.float64).reshape(-1)
        denom = float(np.dot(self.probe_direction, self.probe_direction))
        if denom <= 0:
            return 0.0
        lam = float(np.dot(probe - self.left_probe, self.probe_direction) / denom)
        return float(np.clip(lam, 0.0, 1.0))
    
    def _check_side_of_probe(self, x_lst:list):
        perpToProbeDir = np.array([-self.probe_direction[1], self.probe_direction[0]])
        assert np.isclose(np.dot(self.probe_direction, perpToProbeDir), 0,  rtol=1.e-10, atol=1.e-10)
        side_lst=[]
        for x in x_lst:
            d = x-self.right_probe
            side = np.sign(np.dot(d, perpToProbeDir))
            side_lst.append(side)
        return side_lst


    def _tieline_from_lambda(self, lam):
        """
        Returns phase-labeled interface compositions for one tie-line parameter.

        If a tabulated family of sampled tie-lines is available, this uses
        piecewise linear interpolation through all sampled points. Otherwise it
        falls back to interpolation between the two endpoint tie-lines.
        """
        if (
            self.sampled_lambdas is not None
            and self.sampled_phase_tielines is not None
            and len(self.sampled_lambdas) >= 2
        ):
            matrix_comp = np.array(
                [
                    np.interp(lam, self.sampled_lambdas, self.sampled_phase_tielines[self.matrix_phase][:, i])
                    for i in range(self.sampled_phase_tielines[self.matrix_phase].shape[1])
                ],
                dtype=np.float64,
            )
            precip_comp = np.array(
                [
                    np.interp(lam, self.sampled_lambdas, self.sampled_phase_tielines[self.precip_phase][:, i])
                    for i in range(self.sampled_phase_tielines[self.precip_phase].shape[1])
                ],
                dtype=np.float64,
            )
            return {
                self.matrix_phase: matrix_comp,
                self.precip_phase: precip_comp,
            }
        raise ValueError("Not sure if below stuff is correct!")
        matrix_comp = self.phase_tieline_start[self.matrix_phase] + lam * (
            self.phase_tieline_end[self.matrix_phase] - self.phase_tieline_start[self.matrix_phase]
        )
        precip_comp = self.phase_tieline_start[self.precip_phase] + lam * (
            self.phase_tieline_end[self.precip_phase] - self.phase_tieline_start[self.precip_phase]
        )
        return {
            self.matrix_phase: matrix_comp,
            self.precip_phase: precip_comp,
        }

    def getInterfacialComposition(self, x, T, gExtra=0, precPhase=None, returnMeta=False, xIsLambda=None):
        """
        Returns the independent-component tie-line compositions.

        Parameters
        ----------
        OLD
        x : array-like
            Probe composition(s) in the solver's independent-component
            convention, here ``[x_CR, x_NI]``.
        OLD
        x : float
            Probe lambda
        T, gExtra, precPhase :
            Accepted for API compatibility. This surrogate ignores them.
        returnMeta : bool (optional)
            If True, also returns endpoint metadata with phase labels aligned to
            the returned compositions.
        """
        if xIsLambda is None:
            raise ValueError("This now requires x to be lambda now")
        if (isinstance(x, float)!=True) or (not (0<=x<=1)):
            raise ValueError(f"This now requires x to be lambda now which should be a float between 0 and 1 but for x={x}")
        
        matrix_phase = self.phases[0]
        precip_phase = self.phases[1] if precPhase is None else precPhase
        # if values.ndim == 1:
        if True==True:
            # by_phase = self._tieline_from_lambda(self._lambda_from_probe(values))
            by_phase = self._tieline_from_lambda(x)
            left = np.asarray(by_phase[matrix_phase], dtype=np.float64)
            right = np.asarray(by_phase[precip_phase], dtype=np.float64)
            if not returnMeta:
                return left, right
            return left, right, {
                "endpoint_phases": (matrix_phase, precip_phase),
                "endpoints": (
                    {"phase": matrix_phase, "composition": left},
                    {"phase": precip_phase, "composition": right},
                ),
            }
        # by_phase_pairs = [self._tieline_from_lambda(self._lambda_from_probe(v)) for v in values]
        # left = np.asarray([pair[matrix_phase] for pair in by_phase_pairs], dtype=np.float64)
        # right = np.asarray([pair[precip_phase] for pair in by_phase_pairs], dtype=np.float64)
        # if not returnMeta:
        #     return left, right
        # return left, right, {
        #     "endpoint_phases": (matrix_phase, precip_phase),
        #     "endpoints": (
        #         {"phase": matrix_phase, "composition": left},
        #         {"phase": precip_phase, "composition": right},
        #     ),
        # }

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        """
        Returns interdiffusivity for the requested phase using the configured
        diffusivity surrogate mode.

        In ``nearest_sample`` mode this uses context-aware sample pools:
        - ``query_context='interface'`` searches interface-only samples.
        - ``query_context='general'`` searches combined interface+bulk samples.
        Single-point queries use sum-of-squared-distances, while batched
        queries use sklearn KNN (``k=1``).
        """
        if phase is None:
            if self.diffusivity_mode == "nearest_sample":
                raise ValueError("phase must be specified when diffusivity_mode='nearest_sample'.")
            phase = self.phases[0]
        if phase not in self.diffusivities:
            raise ValueError(f"Unknown phase '{phase}' for diffusivity lookup.")
        base = np.asarray(self.diffusivities[phase], dtype=np.float64)
        values = np.asarray(x, dtype=np.float64)
        if self.diffusivity_mode == "nearest_sample":
            if query_context not in {"interface", "general"}:
                raise ValueError("query_context must be explicitly set to 'interface' or 'general' for nearest_sample diffusivity mode.")
            samples_x = np.asarray(self.sampled_diffusivity_compositions_by_context[query_context][phase], dtype=np.float64)
            samples_d = np.asarray(self.sampled_diffusivities_by_context[query_context][phase], dtype=np.float64)
            nn_model = self._diffusivity_nn_models.get(query_context, {}).get(phase, None)
            if nn_model is None:
                raise ValueError(f"Missing KNN diffusivity index for phase '{phase}' in context '{query_context}'.")
            if values.ndim == 1:
                deltas = samples_x - values.reshape(1, -1)
                indices = np.sum(deltas * deltas, axis=1)
                return np.asarray(samples_d[int(np.argmin(indices))], dtype=np.float64)
            # debugInPlace()
            deltas = values[:, np.newaxis, :] - samples_x[np.newaxis, :, :]
            indices = np.argmin(np.sum(deltas * deltas, axis=2), axis=1)
            ## Overridding the knn because it appears to be slower than squared sums
            # indices = nn_model.kneighbors(values, return_distance=False)
            # indices = indices.reshape(-1)
            return np.asarray(samples_d[indices], dtype=np.float64)
        if values.ndim == 1:
            return base
        return np.tile(base, (len(values), 1, 1))


def _as_independent_components(composition):
    """
    Converts a full ``[x_FE, x_CR, x_NI]`` composition into ``[x_CR, x_NI]``.

    If the composition is already in the independent-component convention used
    by the moving-boundary solver, it is returned unchanged.
    """
    values = np.asarray(composition, dtype=np.float64).reshape(-1)
    if values.size == 2:
        return values
    if values.size == 3:
        return values[1:]
    raise ValueError(
        "Expected either independent ternary components [x_CR, x_NI] or full "
        "substitutional compositions [x_FE, x_CR, x_NI]."
    )


def _valid_tieline(left, right):
    """
    Returns whether a sampled tie-line is usable for the surrogate.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return np.all(np.isfinite(left)) and np.all(np.isfinite(right)) and np.all(left >= 0) and np.all(right >= 0)


def _plot_labeled_ternary_point(ax, independent_composition, label, color, marker="o", text_offset=(0.005, 0.005)):
    """
    Plots one labeled ternary composition point.
    """
    composition = np.asarray(independent_composition, dtype=np.float64).reshape(-1)
    xy = composition[:2]
    ax.scatter(xy[0], xy[1], color=color, marker=marker, s=55, zorder=5)
    ax.text(
        xy[0] + text_offset[0],
        xy[1] + text_offset[1],
        label,
        color=color,
        fontsize=9,
        ha="left",
        va="bottom",
    )
    return xy

def composition_set_to_order(cs, desired_order):
    order = [str(el).upper() for el in cs.phase_record.nonvacant_elements]
    x_by_el = dict(zip(order, np.asarray(cs.X, dtype=np.float64)))
    return np.array([x_by_el[el.upper()] for el in desired_order], dtype=np.float64)

def _phase_compositions_from_workspace(wks):
    """
    Returns phase-labeled independent-component compositions from equilibrium workspace.
    """
    by_phase = {}
    for cs in wks.get_composition_sets():
        phase_name = cs.phase_record.phase_name
        phase_comp = composition_set_to_order(cs, ["CR", "NI"])
        if _valid_tieline(phase_comp, phase_comp):
            by_phase[phase_name] = phase_comp
    return by_phase


def _build_bulk_grid(diffusivity_bulk_bbox, diffusivity_bulk_spacing):
    """
    Builds a Cartesian composition grid for the requested bounding box and spacing.
    """
    bbox = np.asarray(diffusivity_bulk_bbox, dtype=np.float64)
    if bbox.shape != (2, 2):
        raise ValueError("diffusivity_bulk_bbox must have shape (2, 2): [[x0_min, x0_max], [x1_min, x1_max]].")
    spacing = float(diffusivity_bulk_spacing)
    if spacing <= 0:
        raise ValueError("diffusivity_bulk_spacing must be > 0.")
    x0 = np.arange(bbox[0, 0], bbox[0, 1] + 0.5 * spacing, spacing, dtype=np.float64)
    x1 = np.arange(bbox[1, 0], bbox[1, 1] + 0.5 * spacing, spacing, dtype=np.float64)
    if len(x0) == 0 or len(x1) == 0:
        raise ValueError("diffusivity_bulk_bbox and diffusivity_bulk_spacing produced an empty bulk grid.")
    xx0, xx1 = np.meshgrid(x0, x1, indexing="ij")
    return np.column_stack((xx0.reshape(-1), xx1.reshape(-1)))


def _jsonable(value):
    """Converts arrays and paths into JSON-serializable plain values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _build_surrogate_cache_fingerprint(case, seed):
    """Builds deterministic fingerprint payload and SHA256 key for surrogate cache."""
    diffusivity_mode = str(seed.get("diffusivity_mode", "single_tieline"))
    payload = {
        "schema_version": 1,
        "surrogate_cache_version": seed.get("surrogate_cache_version", 0),
        "tdb_path": str(seed["tdb_path"]),
        "temperature": float(seed.get("temperature", case["temperature"])),
        "elements": list(case["elements"]),
        "phases": list(case["phases"]),
        "precipitate_phase": seed.get("precipitate_phase", case["phases"][1]),
        "probe_left": np.asarray(seed["probe_left"], dtype=np.float64),
        "probe_right": np.asarray(seed["probe_right"], dtype=np.float64),
        "probe_lambdas": np.asarray(seed.get("probe_lambdas", np.linspace(0.0, 1.0, 5)), dtype=np.float64),
        "diffusivity_mode": diffusivity_mode,
    }
    if diffusivity_mode == "single_tieline":
        payload["diffusivity_lambda"] = seed.get("diffusivity_lambda", 0.5)
        payload["diffusivity_global"] = None if seed.get("diffusivity_global", None) is None else np.asarray(
            seed.get("diffusivity_global", None), dtype=np.float64
        )
    elif diffusivity_mode == "nearest_sample":
        payload["diffusivity_tieline_lambdas"] = np.asarray(seed.get("diffusivity_tieline_lambdas", []), dtype=np.float64)
        payload["diffusivity_bulk_bbox"] = np.asarray(seed.get("diffusivity_bulk_bbox", []), dtype=np.float64)
        payload["diffusivity_bulk_spacing"] = float(seed.get("diffusivity_bulk_spacing", -1.0))
    else:
        raise ValueError("database_approximation['diffusivity_mode'] must be 'single_tieline' or 'nearest_sample'.")

    payload_json = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    return payload, key


def _get_surrogate_cache_paths(seed, cache_key):
    """Returns canonical npz/json cache paths for this surrogate fingerprint."""
    cache_dir = pathlib.Path(seed.get("surrogate_cache_dir", EXAMPLES_DIR / ".surrogate_cache"))
    stem = f"surrogate_{cache_key[:16]}"
    return {
        "dir": cache_dir,
        "npz": cache_dir / f"{stem}.npz",
        "json": cache_dir / f"{stem}.json",
    }


def _build_database_sampled_tielines(sampled_lambdas, sampled_globals, sampled_phase_tielines, matrix_phase, precip_phase):
    """Rebuilds legacy sampled tie-line list from phase-labeled arrays."""
    sampled = []
    left_arr = np.asarray(sampled_phase_tielines[matrix_phase], dtype=np.float64)
    right_arr = np.asarray(sampled_phase_tielines[precip_phase], dtype=np.float64)
    lam_arr = np.asarray(sampled_lambdas, dtype=np.float64).reshape(-1)
    global_arr = np.asarray(sampled_globals, dtype=np.float64)
    for i in range(len(lam_arr)):
        left_ind = np.asarray(left_arr[i], dtype=np.float64)
        right_ind = np.asarray(right_arr[i], dtype=np.float64)
        sampled.append(
            {
                "lambda": float(lam_arr[i]),
                "global": np.asarray(global_arr[i], dtype=np.float64),
                "phase_ind": {
                    matrix_phase: left_ind.copy(),
                    precip_phase: right_ind.copy(),
                },
                "left_ind": left_ind.copy(),
                "right_ind": right_ind.copy(),
            }
        )
    return sampled


def _save_surrogate_cache(case, cache_paths, cache_payload):
    """Saves surrogate arrays to NPZ and metadata/fingerprint to JSON."""
    matrix_phase = case["phases"][0]
    precip_phase = case["phases"][1]
    arrays = {
        "probe_left": np.asarray(case["probe_left"], dtype=np.float64),
        "probe_right": np.asarray(case["probe_right"], dtype=np.float64),
        "sampled_lambdas": np.asarray(case["sampled_lambdas"], dtype=np.float64),
        "sampled_globals": np.asarray([sample["global"] for sample in case["database_sampled_tielines"]], dtype=np.float64),
        f"phase_tieline_start__{matrix_phase}": np.asarray(case["phase_tieline_start"][matrix_phase], dtype=np.float64),
        f"phase_tieline_start__{precip_phase}": np.asarray(case["phase_tieline_start"][precip_phase], dtype=np.float64),
        f"phase_tieline_end__{matrix_phase}": np.asarray(case["phase_tieline_end"][matrix_phase], dtype=np.float64),
        f"phase_tieline_end__{precip_phase}": np.asarray(case["phase_tieline_end"][precip_phase], dtype=np.float64),
        f"sampled_phase_tielines__{matrix_phase}": np.asarray(case["sampled_phase_tielines"][matrix_phase], dtype=np.float64),
        f"sampled_phase_tielines__{precip_phase}": np.asarray(case["sampled_phase_tielines"][precip_phase], dtype=np.float64),
        f"diffusivities__{matrix_phase}": np.asarray(case["diffusivities"][matrix_phase], dtype=np.float64),
        f"diffusivities__{precip_phase}": np.asarray(case["diffusivities"][precip_phase], dtype=np.float64),
    }

    mode = case["database_diffusivity_mode"]
    if mode == "single_tieline":
        arrays["database_diffusivity_global"] = np.asarray(case["database_diffusivity_global"], dtype=np.float64)
        arrays[f"database_diff_tieline_phase_ind__{matrix_phase}"] = np.asarray(
            case["database_diffusivity_tieline"]["phase_ind"][matrix_phase], dtype=np.float64
        )
        arrays[f"database_diff_tieline_phase_ind__{precip_phase}"] = np.asarray(
            case["database_diffusivity_tieline"]["phase_ind"][precip_phase], dtype=np.float64
        )
    else:
        arrays[f"database_sampled_diff_comp_interface__{matrix_phase}"] = np.asarray(
            case["database_sampled_diffusivity_compositions_interface"][matrix_phase], dtype=np.float64
        )
        arrays[f"database_sampled_diff_comp_interface__{precip_phase}"] = np.asarray(
            case["database_sampled_diffusivity_compositions_interface"][precip_phase], dtype=np.float64
        )
        arrays[f"database_sampled_diff_mats_interface__{matrix_phase}"] = np.asarray(
            case["database_sampled_diffusivities_interface"][matrix_phase], dtype=np.float64
        )
        arrays[f"database_sampled_diff_mats_interface__{precip_phase}"] = np.asarray(
            case["database_sampled_diffusivities_interface"][precip_phase], dtype=np.float64
        )
        arrays[f"database_sampled_diff_comp_general__{matrix_phase}"] = np.asarray(
            case["database_sampled_diffusivity_compositions_general"][matrix_phase], dtype=np.float64
        )
        arrays[f"database_sampled_diff_comp_general__{precip_phase}"] = np.asarray(
            case["database_sampled_diffusivity_compositions_general"][precip_phase], dtype=np.float64
        )
        arrays[f"database_sampled_diff_mats_general__{matrix_phase}"] = np.asarray(
            case["database_sampled_diffusivities_general"][matrix_phase], dtype=np.float64
        )
        arrays[f"database_sampled_diff_mats_general__{precip_phase}"] = np.asarray(
            case["database_sampled_diffusivities_general"][precip_phase], dtype=np.float64
        )
        sampling = case["database_diffusivity_sampling"]
        arrays["database_diff_tieline_lambdas"] = np.asarray(sampling["tieline_lambdas"], dtype=np.float64)
        arrays["database_diff_bulk_bbox"] = np.asarray(sampling["bulk_bbox"], dtype=np.float64)
        arrays["database_diff_bulk_spacing"] = np.asarray([sampling["bulk_spacing"]], dtype=np.float64)

    cache_paths["dir"].mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_paths["npz"], **arrays)
    json_payload = {
        "schema_version": 1,
        "cache_payload": _jsonable(cache_payload),
        "cache_key": cache_paths["npz"].stem.replace("surrogate_", ""),
        "diffusivity_mode": mode,
        "phases": list(case["phases"]),
        "elements": list(case["elements"]),
    }
    cache_paths["json"].write_text(json.dumps(json_payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_surrogate_cache(case, cache_paths, expected_payload):
    """Loads cached surrogate data and repopulates case fields."""
    if not cache_paths["npz"].exists() or not cache_paths["json"].exists():
        return False
    meta = json.loads(cache_paths["json"].read_text(encoding="utf-8"))
    if int(meta.get("schema_version", -1)) != 1:
        return False
    if meta.get("cache_payload", None) != _jsonable(expected_payload):
        return False

    matrix_phase = case["phases"][0]
    precip_phase = case["phases"][1]
    z = np.load(cache_paths["npz"], allow_pickle=False)
    case["probe_left"] = np.asarray(z["probe_left"], dtype=np.float64)
    case["probe_right"] = np.asarray(z["probe_right"], dtype=np.float64)
    case["sampled_lambdas"] = np.asarray(z["sampled_lambdas"], dtype=np.float64)
    sampled_globals = np.asarray(z["sampled_globals"], dtype=np.float64)
    case["phase_tieline_start"] = {
        matrix_phase: np.asarray(z[f"phase_tieline_start__{matrix_phase}"], dtype=np.float64),
        precip_phase: np.asarray(z[f"phase_tieline_start__{precip_phase}"], dtype=np.float64),
    }
    case["phase_tieline_end"] = {
        matrix_phase: np.asarray(z[f"phase_tieline_end__{matrix_phase}"], dtype=np.float64),
        precip_phase: np.asarray(z[f"phase_tieline_end__{precip_phase}"], dtype=np.float64),
    }
    case["sampled_phase_tielines"] = {
        matrix_phase: np.asarray(z[f"sampled_phase_tielines__{matrix_phase}"], dtype=np.float64),
        precip_phase: np.asarray(z[f"sampled_phase_tielines__{precip_phase}"], dtype=np.float64),
    }
    case["left_tieline_start"] = case["phase_tieline_start"][matrix_phase].copy()
    case["left_tieline_end"] = case["phase_tieline_end"][matrix_phase].copy()
    case["right_tieline_start"] = case["phase_tieline_start"][precip_phase].copy()
    case["right_tieline_end"] = case["phase_tieline_end"][precip_phase].copy()
    case["sampled_left_tielines"] = case["sampled_phase_tielines"][matrix_phase]
    case["sampled_right_tielines"] = case["sampled_phase_tielines"][precip_phase]
    case["database_sampled_tielines"] = _build_database_sampled_tielines(
        case["sampled_lambdas"], sampled_globals, case["sampled_phase_tielines"], matrix_phase, precip_phase
    )
    case["database_diffusivity_mode"] = str(meta.get("diffusivity_mode", "single_tieline"))
    case["diffusivities"] = {
        matrix_phase: np.asarray(z[f"diffusivities__{matrix_phase}"], dtype=np.float64),
        precip_phase: np.asarray(z[f"diffusivities__{precip_phase}"], dtype=np.float64),
    }

    if case["database_diffusivity_mode"] == "single_tieline":
        case["database_diffusivity_global"] = np.asarray(z["database_diffusivity_global"], dtype=np.float64)
        left_ind = np.asarray(z[f"database_diff_tieline_phase_ind__{matrix_phase}"], dtype=np.float64)
        right_ind = np.asarray(z[f"database_diff_tieline_phase_ind__{precip_phase}"], dtype=np.float64)
        case["database_diffusivity_tieline"] = {
            "phase_ind": {matrix_phase: left_ind.copy(), precip_phase: right_ind.copy()},
            "phase_full": {matrix_phase: left_ind.copy(), precip_phase: right_ind.copy()},
            "left_ind": left_ind.copy(),
            "right_ind": right_ind.copy(),
            "left_full": left_ind.copy(),
            "right_full": right_ind.copy(),
        }
        case.pop("database_sampled_diffusivity_compositions_interface", None)
        case.pop("database_sampled_diffusivities_interface", None)
        case.pop("database_sampled_diffusivity_compositions_general", None)
        case.pop("database_sampled_diffusivities_general", None)
        case.pop("database_diffusivity_sampling", None)
    else:
        case["database_sampled_diffusivity_compositions_interface"] = {
            matrix_phase: np.asarray(z[f"database_sampled_diff_comp_interface__{matrix_phase}"], dtype=np.float64),
            precip_phase: np.asarray(z[f"database_sampled_diff_comp_interface__{precip_phase}"], dtype=np.float64),
        }
        case["database_sampled_diffusivities_interface"] = {
            matrix_phase: np.asarray(z[f"database_sampled_diff_mats_interface__{matrix_phase}"], dtype=np.float64),
            precip_phase: np.asarray(z[f"database_sampled_diff_mats_interface__{precip_phase}"], dtype=np.float64),
        }
        case["database_sampled_diffusivity_compositions_general"] = {
            matrix_phase: np.asarray(z[f"database_sampled_diff_comp_general__{matrix_phase}"], dtype=np.float64),
            precip_phase: np.asarray(z[f"database_sampled_diff_comp_general__{precip_phase}"], dtype=np.float64),
        }
        case["database_sampled_diffusivities_general"] = {
            matrix_phase: np.asarray(z[f"database_sampled_diff_mats_general__{matrix_phase}"], dtype=np.float64),
            precip_phase: np.asarray(z[f"database_sampled_diff_mats_general__{precip_phase}"], dtype=np.float64),
        }
        case["database_diffusivity_sampling"] = {
            "tieline_lambdas": np.asarray(z["database_diff_tieline_lambdas"], dtype=np.float64),
            "bulk_bbox": np.asarray(z["database_diff_bulk_bbox"], dtype=np.float64),
            "bulk_spacing": float(np.asarray(z["database_diff_bulk_spacing"], dtype=np.float64).reshape(-1)[0]),
            "sample_count_by_phase": {
                matrix_phase: int(case["database_sampled_diffusivity_compositions_general"][matrix_phase].shape[0]),
                precip_phase: int(case["database_sampled_diffusivity_compositions_general"][precip_phase].shape[0]),
            },
            "sample_count_by_phase_interface": {
                matrix_phase: int(case["database_sampled_diffusivity_compositions_interface"][matrix_phase].shape[0]),
                precip_phase: int(case["database_sampled_diffusivity_compositions_interface"][precip_phase].shape[0]),
            },
        }
        case.pop("database_diffusivity_global", None)
        case.pop("database_diffusivity_tieline", None)

    return True


def build_database_seed_thermodynamics(case):
    """
    Creates the real thermodynamics object used only for seeding the surrogate.
    """
    seed = case["database_approximation"]
    return MulticomponentThermodynamics(
        str(seed["tdb_path"]),
        case["elements"],
        seed.get("phases", case["phases"]),
    )


def apply_database_approximation(case):
    """
    Replaces surrogate tie-lines and diffusivities with values sampled once
    from the Lee 1993 mobility database.

    The tie-line family is approximated by sampling several global
    compositions along a user-defined probe line in the alpha-gamma two-phase
    region at the requested temperature. The surrogate then uses the first and
    last valid sampled tie-lines as its linear endpoints.

    Diffusivity surrogates support two modes:
    - ``single_tieline``: one frozen matrix per phase from a single tie-line.
    - ``nearest_sample``: per-phase sampled matrices returned by nearest
      composition in independent-component space.
    """
    seed = case.get("database_approximation", {})
    if not seed.get("enabled", False):
        return case

    cache_enabled = bool(seed.get("surrogate_cache_enabled", True))
    cache_policy = str(seed.get("surrogate_cache_policy", "load_or_build"))
    if cache_policy not in {"load_or_build", "load_only", "build_only"}:
        raise ValueError("database_approximation['surrogate_cache_policy'] must be 'load_or_build', 'load_only', or 'build_only'.")
    cache_payload, cache_key = _build_surrogate_cache_fingerprint(case, seed)
    cache_paths = _get_surrogate_cache_paths(seed, cache_key)
    case["database_surrogate_cache_key"] = cache_key[:16]
    case["database_surrogate_cache_paths"] = {
        "npz": str(cache_paths["npz"]),
        "json": str(cache_paths["json"]),
    }
    case["database_surrogate_cache_hit"] = False
    case.pop("database_diffusivity_surrogate_load_time_s", None)
    if cache_enabled and cache_policy != "build_only":
        t0_load = time.perf_counter()
        try:
            cache_hit = _load_surrogate_cache(case, cache_paths, cache_payload)
        except Exception:
            cache_hit = False
        if cache_hit:
            case["database_surrogate_cache_hit"] = True
            case["database_diffusivity_surrogate_load_time_s"] = float(time.perf_counter() - t0_load)
            case["database_diffusivity_surrogate_build_time_s"] = 0.0
            print(
                "Loaded surrogate cache "
                f"{case['database_surrogate_cache_key']} in {case['database_diffusivity_surrogate_load_time_s']:.3f} s"
            )
            return case
        if cache_policy == "load_only":
            raise ValueError(
                "Surrogate cache miss (or invalid cache) with surrogate_cache_policy='load_only'. "
                f"Expected files: {cache_paths['npz']} and {cache_paths['json']}"
            )

    therm = build_database_seed_thermodynamics(case)
    temperature = float(seed.get("temperature", case["temperature"]))
    probe_left = np.asarray(seed["probe_left"], dtype=np.float64)
    probe_right = np.asarray(seed["probe_right"], dtype=np.float64)
    probe_lambdas = np.asarray(seed.get("probe_lambdas", np.linspace(0.0, 1.0, 5)), dtype=np.float64)
    prec_phase = seed.get("precipitate_phase", case["phases"][1])

    sampled = []
    matrix_phase = case["phases"][0]
    precip_phase = prec_phase
    for lam in probe_lambdas:
        global_composition = (1.0 - lam) * probe_left + lam * probe_right
        c0_full, c1_full, meta = therm.getInterfacialComposition(
            global_composition,
            temperature,
            precPhase=prec_phase,
            returnMeta=True,
        )
        endpoints = meta.get("endpoints", None) if isinstance(meta, dict) else None
        if endpoints is None or len(endpoints) != 2:
            continue
        by_phase_full = {
            endpoints[0]["phase"]: np.asarray(endpoints[0]["composition"], dtype=np.float64),
            endpoints[1]["phase"]: np.asarray(endpoints[1]["composition"], dtype=np.float64),
        }
        if matrix_phase not in by_phase_full or precip_phase not in by_phase_full:
            continue
        matrix_ind = _as_independent_components(by_phase_full[matrix_phase])
        precip_ind = _as_independent_components(by_phase_full[precip_phase])
        if _valid_tieline(matrix_ind, precip_ind):
            sampled.append(
                {
                    "lambda": float(lam),
                    "global": global_composition,
                    "phase_full": by_phase_full,
                    "phase_ind": {
                        matrix_phase: matrix_ind,
                        precip_phase: precip_ind,
                    },
                    # Legacy aliases retained for existing plotting/debug code.
                    "left_full": np.asarray(by_phase_full[matrix_phase], dtype=np.float64),
                    "right_full": np.asarray(by_phase_full[precip_phase], dtype=np.float64),
                    "left_ind": matrix_ind,
                    "right_ind": precip_ind,
                }
            )

    if len(sampled) < 2:
        raise ValueError(
            "Database approximation could not find at least two valid alpha-gamma tie-lines. "
            "Adjust database_approximation['probe_left'] and ['probe_right'] to stay inside the "
            "1373 K two-phase region."
        )
    # Validate that entered phase ordering matches the profile-side chemistry.
    # We expect the left bulk state to be closer to the phase listed first in
    # case["phases"] (matrix side) and the right bulk state to be closer to the
    # precipitate phase. If this is not true for most sampled tie-lines, fail
    # loudly because the input phase order is likely reversed.
    left_bulk = np.asarray(case["left_bulk"], dtype=np.float64).reshape(-1)
    right_bulk = np.asarray(case["right_bulk"], dtype=np.float64).reshape(-1)
    side_match_count = 0
    for sample in sampled:
        matrix_comp = np.asarray(sample["phase_ind"][matrix_phase], dtype=np.float64).reshape(-1)
        precip_comp = np.asarray(sample["phase_ind"][precip_phase], dtype=np.float64).reshape(-1)
        d_left_matrix = float(np.linalg.norm(left_bulk - matrix_comp))
        d_left_precip = float(np.linalg.norm(left_bulk - precip_comp))
        d_right_matrix = float(np.linalg.norm(right_bulk - matrix_comp))
        d_right_precip = float(np.linalg.norm(right_bulk - precip_comp))
        left_matches = d_left_matrix <= d_left_precip
        right_matches = d_right_precip <= d_right_matrix
        if left_matches and right_matches:
            side_match_count += 1
    if side_match_count < max(1, int(np.ceil(0.5 * len(sampled)))):
        raise ValueError(
            "Phase-order validation failed: left/right bulk compositions do not match the entered "
            f"phase ordering {case['phases']} for most sampled tie-lines. "
            f"Try swapping phase order to {[case['phases'][1], case['phases'][0]]}. "
            f"Matches: {side_match_count}/{len(sampled)}."
        )

    tie_start = sampled[0]
    tie_end = sampled[-1]
    diffusivity_mode = str(seed.get("diffusivity_mode", "single_tieline"))
    if diffusivity_mode not in {"single_tieline", "nearest_sample"}:
        raise ValueError("database_approximation['diffusivity_mode'] must be 'single_tieline' or 'nearest_sample'.")

    case["temperature"] = temperature
    case["probe_left"] = probe_left.copy()
    case["probe_right"] = probe_right.copy()
    case["phase_tieline_start"] = {
        matrix_phase: tie_start["phase_ind"][matrix_phase].copy(),
        precip_phase: tie_start["phase_ind"][precip_phase].copy(),
    }
    case["phase_tieline_end"] = {
        matrix_phase: tie_end["phase_ind"][matrix_phase].copy(),
        precip_phase: tie_end["phase_ind"][precip_phase].copy(),
    }
    # Legacy aliases retained for existing plotting/debug code in this file.
    case["left_tieline_start"] = case["phase_tieline_start"][matrix_phase].copy()
    case["left_tieline_end"] = case["phase_tieline_end"][matrix_phase].copy()
    case["right_tieline_start"] = case["phase_tieline_start"][precip_phase].copy()
    case["right_tieline_end"] = case["phase_tieline_end"][precip_phase].copy()
    case["database_diffusivity_mode"] = diffusivity_mode

    diffusivity_build_t0 = time.perf_counter()
    if diffusivity_mode == "single_tieline":
        explicit_diffusivity_global = seed.get("diffusivity_global", None)
        if explicit_diffusivity_global is None:
            diffusivity_lambda = float(seed.get("diffusivity_lambda", 0.5))
            diffusivity_global = (1.0 - diffusivity_lambda) * probe_left + diffusivity_lambda * probe_right
        else:
            diffusivity_global = np.asarray(explicit_diffusivity_global, dtype=np.float64)
        _, _, diff_meta = therm.getInterfacialComposition(
            diffusivity_global,
            temperature,
            precPhase=prec_phase,
            returnMeta=True,
        )
        diff_endpoints = diff_meta.get("endpoints", None) if isinstance(diff_meta, dict) else None
        if diff_endpoints is None or len(diff_endpoints) != 2:
            raise ValueError("Database approximation could not resolve phase-labeled diffusivity tie-line endpoints.")
        diff_by_phase_full = {
            diff_endpoints[0]["phase"]: np.asarray(diff_endpoints[0]["composition"], dtype=np.float64),
            diff_endpoints[1]["phase"]: np.asarray(diff_endpoints[1]["composition"], dtype=np.float64),
        }
        if matrix_phase not in diff_by_phase_full or precip_phase not in diff_by_phase_full:
            raise ValueError(
                f"Database approximation tie-line metadata did not include required phases ({matrix_phase}, {precip_phase})."
            )
        diff_matrix_ind = _as_independent_components(diff_by_phase_full[matrix_phase])
        diff_precip_ind = _as_independent_components(diff_by_phase_full[precip_phase])
        if not _valid_tieline(diff_matrix_ind, diff_precip_ind):
            raise ValueError(
                "Database approximation could not evaluate a valid tie-line at diffusivity_lambda. "
                "Choose a diffusivity_lambda that lies within the sampled two-phase interval."
            )
        case["diffusivities"] = {
            case["phases"][0]: np.asarray(
                therm.getInterdiffusivity(diff_matrix_ind, temperature, phase=case["phases"][0]),
                dtype=np.float64,
            ),
            case["phases"][1]: np.asarray(
                therm.getInterdiffusivity(diff_precip_ind, temperature, phase=case["phases"][1]),
                dtype=np.float64,
            ),
        }
        case["database_diffusivity_global"] = diffusivity_global
        case["database_diffusivity_tieline"] = {
            "phase_ind": {
                matrix_phase: diff_matrix_ind.copy(),
                precip_phase: diff_precip_ind.copy(),
            },
            "phase_full": {
                matrix_phase: np.asarray(diff_by_phase_full[matrix_phase], dtype=np.float64),
                precip_phase: np.asarray(diff_by_phase_full[precip_phase], dtype=np.float64),
            },
            "left_ind": diff_matrix_ind.copy(),
            "right_ind": diff_precip_ind.copy(),
            "left_full": np.asarray(diff_by_phase_full[matrix_phase], dtype=np.float64),
            "right_full": np.asarray(diff_by_phase_full[precip_phase], dtype=np.float64),
        }
    else:
        diffusivity_tieline_lambdas = np.asarray(seed.get("diffusivity_tieline_lambdas", []), dtype=np.float64).reshape(-1)
        if diffusivity_tieline_lambdas.size == 0:
            raise ValueError(
                "database_approximation['diffusivity_tieline_lambdas'] must be provided for diffusivity_mode='nearest_sample'."
            )
        if "diffusivity_bulk_bbox" not in seed or "diffusivity_bulk_spacing" not in seed:
            raise ValueError(
                "database_approximation must include 'diffusivity_bulk_bbox' and 'diffusivity_bulk_spacing' for diffusivity_mode='nearest_sample'."
            )
        sampled_diffusivity_points_interface = {matrix_phase: [], precip_phase: []}
        sampled_diffusivity_mats_interface = {matrix_phase: [], precip_phase: []}
        sampled_diffusivity_points_general = {matrix_phase: [], precip_phase: []}
        sampled_diffusivity_mats_general = {matrix_phase: [], precip_phase: []}

        for lam in diffusivity_tieline_lambdas:
            global_composition = (1.0 - lam) * probe_left + lam * probe_right
            _, _, diff_meta = therm.getInterfacialComposition(
                global_composition,
                temperature,
                precPhase=prec_phase,
                returnMeta=True,
            )
            endpoints = diff_meta.get("endpoints", None) if isinstance(diff_meta, dict) else None
            if endpoints is None or len(endpoints) != 2:
                raise ValueError(
                    f"Could not resolve diffusivity tie-line endpoints at lambda={float(lam):.6g}."
                )
            phase_seen = set()
            for endpoint in endpoints:
                phase_name = endpoint["phase"]
                if phase_name not in sampled_diffusivity_points_interface:
                    continue
                phase_ind = _as_independent_components(np.asarray(endpoint["composition"], dtype=np.float64))
                if not np.all(np.isfinite(phase_ind)):
                    raise ValueError(
                        f"Non-finite diffusivity tie-line endpoint composition at lambda={float(lam):.6g} for phase {phase_name}."
                    )
                dmat = np.asarray(therm.getInterdiffusivity(phase_ind, temperature, phase=phase_name), dtype=np.float64)
                sampled_diffusivity_points_interface[phase_name].append(phase_ind)
                sampled_diffusivity_mats_interface[phase_name].append(dmat)
                sampled_diffusivity_points_general[phase_name].append(phase_ind)
                sampled_diffusivity_mats_general[phase_name].append(dmat)
                phase_seen.add(phase_name)
            missing = [p for p in [matrix_phase, precip_phase] if p not in phase_seen]
            if missing:
                raise ValueError(
                    f"Diffusivity tie-line at lambda={float(lam):.6g} did not include required phases: {missing}."
                )

        bulk_grid = _build_bulk_grid(seed["diffusivity_bulk_bbox"], seed["diffusivity_bulk_spacing"])
        for global_composition in bulk_grid:
            try:
                wks = therm.getEq(global_composition, temperature, 0, [matrix_phase, precip_phase])
                # debugInPlace()
                by_phase_ind = _phase_compositions_from_workspace(wks)
            except Exception:
                # debugInPlace()
                raise
                continue
            for phase_name, phase_ind in by_phase_ind.items():
                if phase_name not in sampled_diffusivity_points_general:
                    continue
                sampled_diffusivity_points_general[phase_name].append(np.asarray(phase_ind, dtype=np.float64))
                sampled_diffusivity_mats_general[phase_name].append(
                    np.asarray(therm.getInterdiffusivity(phase_ind, temperature, phase=phase_name), dtype=np.float64)
                )

        sampled_comp_out_interface = {}
        sampled_mat_out_interface = {}
        sampled_comp_out_general = {}
        sampled_mat_out_general = {}
        for phase_name in [matrix_phase, precip_phase]:
            if len(sampled_diffusivity_points_interface[phase_name]) == 0:
                raise ValueError(
                    f"No interface sampled diffusivity points were collected for phase {phase_name} in nearest_sample mode."
                )
            if len(sampled_diffusivity_points_general[phase_name]) == 0:
                raise ValueError(
                    f"No general sampled diffusivity points were collected for phase {phase_name} in nearest_sample mode."
                )
            sampled_comp_out_interface[phase_name] = np.asarray(sampled_diffusivity_points_interface[phase_name], dtype=np.float64)
            sampled_mat_out_interface[phase_name] = np.asarray(sampled_diffusivity_mats_interface[phase_name], dtype=np.float64)
            sampled_comp_out_general[phase_name] = np.asarray(sampled_diffusivity_points_general[phase_name], dtype=np.float64)
            sampled_mat_out_general[phase_name] = np.asarray(sampled_diffusivity_mats_general[phase_name], dtype=np.float64)

        case["database_sampled_diffusivity_compositions_interface"] = sampled_comp_out_interface
        case["database_sampled_diffusivities_interface"] = sampled_mat_out_interface
        case["database_sampled_diffusivity_compositions_general"] = sampled_comp_out_general
        case["database_sampled_diffusivities_general"] = sampled_mat_out_general
        case["database_diffusivity_sampling"] = {
            "tieline_lambdas": diffusivity_tieline_lambdas.copy(),
            "bulk_bbox": np.asarray(seed["diffusivity_bulk_bbox"], dtype=np.float64),
            "bulk_spacing": float(seed["diffusivity_bulk_spacing"]),
            "sample_count_by_phase": {
                matrix_phase: int(sampled_comp_out_general[matrix_phase].shape[0]),
                precip_phase: int(sampled_comp_out_general[precip_phase].shape[0]),
            },
            "sample_count_by_phase_interface": {
                matrix_phase: int(sampled_comp_out_interface[matrix_phase].shape[0]),
                precip_phase: int(sampled_comp_out_interface[precip_phase].shape[0]),
            },
        }

        case["diffusivities"] = {
            phase_name: np.asarray(sampled_mat_out_general[phase_name][0], dtype=np.float64)
            for phase_name in [matrix_phase, precip_phase]
        }
    case["database_diffusivity_surrogate_build_time_s"] = float(time.perf_counter() - diffusivity_build_t0)
    print(
        "Diffusivity surrogate build time "
        f"({diffusivity_mode}) = {case['database_diffusivity_surrogate_build_time_s']:.3f} s"
    )
    case["database_sampled_tielines"] = sampled
    case["sampled_lambdas"] = np.asarray([sample["lambda"] for sample in sampled], dtype=np.float64)
    case["sampled_phase_tielines"] = {
        matrix_phase: np.asarray([sample["phase_ind"][matrix_phase] for sample in sampled], dtype=np.float64),
        precip_phase: np.asarray([sample["phase_ind"][precip_phase] for sample in sampled], dtype=np.float64),
    }
    case["sampled_left_tielines"] = case["sampled_phase_tielines"][matrix_phase]
    case["sampled_right_tielines"] = case["sampled_phase_tielines"][precip_phase]
    if cache_enabled:
        _save_surrogate_cache(case, cache_paths, cache_payload)
    return case


def build_ternary_case():
    """
    Returns a lightweight ternary moving-boundary setup using a proven-stable
    synthetic tie-line family that exercises the ternary interface solver
    without database calls.

    The defaults below are intentionally close in flavor to Fe-Cr-Ni
    partitioning, but they are primarily chosen to give a robust sandbox for
    debugging the moving-boundary algorithm. As you start matching a real
    database more closely, the main values to tune are:

    - ``left_bulk`` / ``right_bulk``
    - ``left_tieline_start`` / ``left_tieline_end``
    - ``right_tieline_start`` / ``right_tieline_end``
    - ``diffusivities``
    """
    case = {
        "elements": ["FE", "CR", "NI"],
        "independent_elements": ["CR", "NI"],
        "phases": ["BCC_A2", "FCC_A1"],
        "temperature": 1373.0,
        "length": [1.0, 30e-6, 30e-6, 62.05e-6+2000e-6][2],
        "nodes": [30+1, 1000+1, 60+1][0],
        "interface_position": [0.515, 18e-6, 18e-6+1e-8, 62.05e-6, 12e-6+1e-12, 12e-6-1e-9][-2], 
        "left_bulk": [np.array([0.38, 0.001], dtype=np.float64), np.array([0.39696, 0.0001], dtype=np.float64)][0],
        "right_bulk": [np.array([0.13, 0.15], dtype=np.float64), np.array([0.139297, 0.142387], dtype=np.float64)][0],
        "probe_left": np.array([0.1233, 0.0001], dtype=np.float64),
        "probe_right": np.array([0.4993, 0.2257], dtype=np.float64), # np.array([0.3113, 0.1129], dtype=np.float64),
        "left_tieline_start": None, #np.array([0.286, 0.0575], dtype=np.float64),
        "left_tieline_end": None, #np.array([0.246, 0.0675], dtype=np.float64),
        "right_tieline_start": None, #np.array([0.188, 0.12875], dtype=np.float64),
        "right_tieline_end": None, #np.array([0.218, 0.16875], dtype=np.float64),
        "diffusivities": {
            "FCC_A1": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64),
            "BCC_A2": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64),
        },
        "solve_time": [5.0e-5, 3600*1e2][1],
        "record": True,
        "vIt": 5000,
        "moving_boundary_threshold": (min(pstar_input, (1-(2.0*pstar_input))) if pstar_input<0.5 else pstar_input)*0.98,
        "database_approximation": {
            "enabled": True,
            "tdb_path": EXAMPLES_DIR / "FeCrNi_Lee1993_L_style_ternary_checked_withMobility.tdb",
            "temperature": 1373.0,
            "phases": ["BCC_A2", "FCC_A1"],
            "precipitate_phase": "FCC_A1",
            "probe_left": np.array([0.1233, 0.0001], dtype=np.float64), # np.array([0.26, 0.0575], dtype=np.float64),
            "probe_right": np.array([0.4993, 0.2257], dtype=np.float64), #np.array([0.3113, 0.1129], dtype=np.float64), # np.array([0.25, 0.08], dtype=np.float64),
            "probe_lambdas": np.linspace(0.0, 1.0, 50),
            "diffusivity_mode": "nearest_sample", # "single_tieline", "nearest_sample"
            "diffusivity_lambda": 0.5,
            # Used when diffusivity_mode == "nearest_sample":
            "diffusivity_tieline_lambdas": np.linspace(0.0, 1.0, 25),
            "diffusivity_bulk_bbox": np.array([[0.001, 0.45], [0.001, 0.20]], dtype=np.float64),
            "diffusivity_bulk_spacing": 0.02,
            "surrogate_cache_enabled": True,
            "surrogate_cache_dir": EXAMPLES_DIR / ".surrogate_cache",
            "surrogate_cache_policy": "load_or_build",  # "load_or_build", "load_only", "build_only"
            "surrogate_cache_version": 0,
            # Optional alternative to diffusivity_lambda:
            # "diffusivity_global": np.array([0.255, 0.06875], dtype=np.float64),
        },
        "actual_database_models": {
            "enabled": True,
            "model_specs": {
                # "actual_basic": {
                #     "interface_update": "basic",
                #     "balance_element": "CR",
                # },
                # "actual_corrected_CR": {
                #     "interface_update": "lee_oh_corrected",
                #     "balance_element": "CR",
                # },
                # "actual_my_corrected_CR": {
                #     "interface_update": "my_corrected",
                #     "balance_element": "CR",
                # },
                # "actual_corrected_NI": {
                #     "interface_update": "lee_oh_corrected",
                #     "balance_element": "NI",
                # },
            },
        },
    }
    return apply_database_approximation(case)


def make_mesh(case):
    """Builds the ternary step profile used as the initial condition."""
    profile = ProfileBuilder(
        [
            (StepProfile1D(case["interface_position"], case["left_bulk"][0], case["right_bulk"][0]), "CR"),
            (StepProfile1D(case["interface_position"], case["left_bulk"][1], case["right_bulk"][1]), "NI"),
        ]
    )
    mesh = CartesianFD1D(case["independent_elements"], [0.0, case["length"]], case["nodes"])
    mesh.setResponseProfile(profile)
    return mesh


def make_surrogate_thermodynamics(case):
    """Creates the configurable surrogate thermodynamics object."""
    return ApproximateTernaryTieLineThermodynamics(
        phases=case["phases"],
        left_probe=case.get("probe_left", case["left_bulk"]),
        right_probe=case.get("probe_right", case["right_bulk"]),
        left_tieline_start=case["left_tieline_start"],
        left_tieline_end=case["left_tieline_end"],
        right_tieline_start=case["right_tieline_start"],
        right_tieline_end=case["right_tieline_end"],
        diffusivities=case["diffusivities"],
        sampled_lambdas=case.get("sampled_lambdas"),
        sampled_left_tielines=case.get("sampled_left_tielines"),
        sampled_right_tielines=case.get("sampled_right_tielines"),
        phase_tieline_start=case.get("phase_tieline_start"),
        phase_tieline_end=case.get("phase_tieline_end"),
        sampled_phase_tielines=case.get("sampled_phase_tielines"),
        diffusivity_mode=case.get("database_diffusivity_mode", case.get("database_approximation", {}).get("diffusivity_mode", "single_tieline")),
        sampled_diffusivity_compositions=case.get("database_sampled_diffusivity_compositions_general"),
        sampled_diffusivities=case.get("database_sampled_diffusivities_general"),
        sampled_diffusivity_tieline_lambdas=None if case.get("database_diffusivity_sampling") is None else case["database_diffusivity_sampling"].get("tieline_lambdas"),
        tdb_ForPlotting=case["database_approximation"]["tdb_path"] if case.get("database_approximation", {}).get("enabled", False) else None,
    )


def make_actual_thermodynamics(case):
    """
    Creates the real thermodynamics / kinetics object from the Lee 1993 TDB.
    """
    seed = case["database_approximation"]
    return MulticomponentThermodynamics(
        str(seed["tdb_path"]),
        case["elements"],
        seed.get("phases", case["phases"]),
    )


def plot_ternary_thermodynamics(case):
    """
    Plots the sampled database tie-lines against the surrogate tie-line family
    on ordinary Cartesian axes using ``X(CR)`` vs ``X(NI)``.

    This keeps all thermodynamics information on one plot while preserving
    matplotlib's normal zoom and pan behavior.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    # fig = plt.figure(figsize=(9, 7))
    # ax = fig.add_subplot(projection='triangular')

    seed = case["database_approximation"]
    dbf = Database(str(seed["tdb_path"]))
    comps = case["elements"] + ["VA"]
    phases = seed.get("phases", case["phases"])
    conds = {
        v.T: float(seed.get("temperature", case["temperature"])),
        v.P: 101325,
        v.X(case["independent_elements"][0]): (0, 1, 0.01),
        v.X(case["independent_elements"][1]): (0, 1, 0.01),
    }
    ternplot(
        dbf,
        comps,
        phases,
        conds,
        x=v.X(case["independent_elements"][0]),
        y=v.X(case["independent_elements"][1]),
        ax=ax,
        tielines=False,
        s=4,
        alpha=0.35,
    )

    surrogate = make_surrogate_thermodynamics(case)
    surrogate_lambdas = np.linspace(0.0, 1.0, 51)
    probe_left = np.asarray(case.get("probe_left", case["left_bulk"]), dtype=np.float64)
    probe_right = np.asarray(case.get("probe_right", case["right_bulk"]), dtype=np.float64)
    surrogate_left_points = []
    surrogate_right_points = []
    for lam in surrogate_lambdas:
        probe = (1.0 - lam) * probe_left + lam * probe_right
        left_ind, right_ind = surrogate.getInterfacialComposition(probe, case["temperature"])
        surrogate_left_points.append(np.asarray(left_ind, dtype=np.float64))
        surrogate_right_points.append(np.asarray(right_ind, dtype=np.float64))
    surrogate_left_points = np.asarray(surrogate_left_points, dtype=np.float64)
    surrogate_right_points = np.asarray(surrogate_right_points, dtype=np.float64)
    ax.plot(
        surrogate_left_points[:, 0],
        surrogate_left_points[:, 1],
        color="tab:blue",
        linewidth=2.0,
        label="Approx. left phase boundary",
    )
    ax.plot(
        surrogate_right_points[:, 0],
        surrogate_right_points[:, 1],
        color="tab:orange",
        linewidth=2.0,
        label="Approx. right phase boundary",
    )
    for lam in np.linspace(0.0, 1.0, 5):
        probe = (1.0 - lam) * probe_left + lam * probe_right
        left_ind, right_ind = surrogate.getInterfacialComposition(probe, case["temperature"])
        left_xy = np.asarray(left_ind, dtype=np.float64)
        right_xy = np.asarray(right_ind, dtype=np.float64)
        ax.plot(
            [left_xy[0], right_xy[0]],
            [left_xy[1], right_xy[1]],
            color="tab:purple",
            linewidth=1.0,
            alpha=0.45,
        )

    sampled = case.get("database_sampled_tielines", [])
    if sampled:
        true_left_points = np.asarray(
            [sample["left_ind"] for sample in sampled],
            dtype=np.float64,
        )
        true_right_points = np.asarray(
            [sample["right_ind"] for sample in sampled],
            dtype=np.float64,
        )
        ax.plot(
            true_left_points[:, 0],
            true_left_points[:, 1],
            color="tab:green",
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="TDB left phase boundary",
        )
        ax.plot(
            true_right_points[:, 0],
            true_right_points[:, 1],
            color="tab:red",
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="TDB right phase boundary",
        )
        for sample in sampled:
            left_xy = np.asarray(sample["left_ind"], dtype=np.float64)
            right_xy = np.asarray(sample["right_ind"], dtype=np.float64)
            ax.plot(
                [left_xy[0], right_xy[0]],
                [left_xy[1], right_xy[1]],
                color="0.45",
                linewidth=1.0,
                linestyle="--",
                alpha=0.7,
            )
            global_xy = np.asarray(sample["global"], dtype=np.float64)
            ax.scatter(global_xy[0], global_xy[1], color="0.35", marker="x", s=35, zorder=4)

    _plot_labeled_ternary_point(ax, case["left_bulk"], "left_bulk", color="black", marker="s", text_offset=(-0.035, 0.004))
    _plot_labeled_ternary_point(ax, case["right_bulk"], "right_bulk", color="black", marker="s", text_offset=(0.008, -0.008))
    _plot_labeled_ternary_point(ax, probe_left, "probe_left", color="tab:blue", marker="^", text_offset=(-0.03, 0.006))
    _plot_labeled_ternary_point(ax, probe_right, "probe_right", color="tab:orange", marker="^", text_offset=(0.008, 0.006))
    _plot_labeled_ternary_point(ax, case["left_tieline_start"], "approx_left_start", color="tab:blue", marker="o", text_offset=(-0.045, 0.006))
    _plot_labeled_ternary_point(ax, case["left_tieline_end"], "approx_left_end", color="tab:blue", marker="o", text_offset=(-0.04, 0.0))
    _plot_labeled_ternary_point(ax, case["right_tieline_start"], "approx_right_start", color="tab:orange", marker="o", text_offset=(0.008, -0.008))
    _plot_labeled_ternary_point(ax, case["right_tieline_end"], "approx_right_end", color="tab:orange", marker="o", text_offset=(0.008, 0.006))

    if "database_diffusivity_global" in case:
        _plot_labeled_ternary_point(
            ax,
            case["database_diffusivity_global"],
            "diffusivity_global",
            color="tab:green",
            marker="D",
            text_offset=(0.008, 0.006),
        )
    if "database_diffusivity_tieline" in case:
        _plot_labeled_ternary_point(
            ax,
            case["database_diffusivity_tieline"]["left_ind"],
            "tdb_diff_left",
            color="tab:green",
            marker="D",
            text_offset=(-0.035, 0.006),
        )
        _plot_labeled_ternary_point(
            ax,
            case["database_diffusivity_tieline"]["right_ind"],
            "tdb_diff_right",
            color="tab:red",
            marker="D",
            text_offset=(0.008, -0.008),
        )

    all_points = [
        surrogate_left_points,
        surrogate_right_points,
        np.asarray([probe_left, probe_right, case["left_bulk"], case["right_bulk"]], dtype=np.float64),
    ]
    if sampled:
        all_points.extend([true_left_points, true_right_points, np.asarray([sample["global"] for sample in sampled], dtype=np.float64)])
    if "database_diffusivity_global" in case:
        all_points.append(np.asarray([case["database_diffusivity_global"]], dtype=np.float64))
    if "database_diffusivity_tieline" in case:
        all_points.append(
            np.asarray(
                [
                    case["database_diffusivity_tieline"]["left_ind"],
                    case["database_diffusivity_tieline"]["right_ind"],
                ],
                dtype=np.float64,
            )
        )
    all_points = np.vstack(all_points)
    x_values = all_points[:, 0]
    y_values = all_points[:, 1]
    x_pad = max(0.01, 0.08 * (np.max(x_values) - np.min(x_values)))
    y_pad = max(0.01, 0.08 * (np.max(y_values) - np.min(y_values)))
    # ax.set_xlim(max(0.0, np.min(x_values) - x_pad), min(1.0, np.max(x_values) + x_pad))
    # ax.set_ylim(max(0.0, np.min(y_values) - y_pad), min(1.0, np.max(y_values) + y_pad))

    ax.set_xlabel(f"X({case['independent_elements'][0]})")
    ax.set_ylabel(f"X({case['independent_elements'][1]})")
    ax.set_title("Thermodynamics: TDB vs Approximation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    if plt.get_backend().lower() != "agg":
        plt.show()
    return fig, ax


def build_model(case, interface_update, balance_element):
    """Constructs one ternary moving-boundary FDM model variant."""
    return _build_model(case, interface_update, balance_element, use_surrogate=True)


def _build_model(case, interface_update, balance_element, use_surrogate=True):
    """
    Constructs either a surrogate-backed or database-backed ternary model.
    """
    constraints = DiffusionConstraints()
    constraints.movingBoundaryThreshold = float(case.get("moving_boundary_threshold", constraints.movingBoundaryThreshold))
    thermodynamics = make_surrogate_thermodynamics(case) if use_surrogate else make_actual_thermodynamics(case)
    thermodynamics.left_probe = case["probe_left"]
    thermodynamics.right_probe = case["probe_right"]

    model = MovingBoundaryFD1DModel(
        make_mesh(case),
        case["elements"],
        case["phases"],
        thermodynamics=thermodynamics,
        temperature=TemperatureParameters(case["temperature"]),
        interfacePosition=case["interface_position"],
        bulkUpdateScheme="flux_form",
        integrationMode= "noIgnore", #"noIgnore", "ignore", "weighted",
        ignoredNodeRule="lee_oh_1996_three_region", # "legacy_two_region", "lee_oh_1996_three_region"
        denom_type="eqn22", # "eqn22", "eqn11"
        ignoredNodeReconstructionMode="lagrange", # "lagrange", "linear"
        initialInventoryMode="integrated", # "phase_length_idealized", "integrated"
        interfaceUpdate=interface_update,
        pstar=pstar_input,
        fluxGradientMode="pre_diffusion", #"pre_diffusion", "post_diffusion"
        balanceElement=balance_element,
        multicomponentInterfaceStateUpdate="pre_diffusion_only", #"pre_diffusion_only", "pre_and_post_diffusion",
        constraints=constraints,
        record=case["record"],
    )
    model.setup()
    return model


def solve_models(case):
    """Runs a few model variants so the ternary moving-boundary logic is easy to compare."""
    model_specs = {
        # "basic": {
        #     "interface_update": "basic",
        #     "balance_element": "CR",
        #     "use_surrogate": True,
        # },
        "corrected_CR": {
            "interface_update": "lee_oh_corrected",
            "balance_element": "CR",
            "use_surrogate": True,
        },
        # "corrected_NI": {
        #     "interface_update": "lee_oh_corrected",
        #     "balance_element": "NI",
        #     "use_surrogate": True,
        # },
        # "corrected_my_CR": {
        #     "interface_update": "my_corrected",
        #     "balance_element": "CR",
        #     "use_surrogate": True,
        # },
        # "corrected_my_NI": {
        #     "interface_update": "my_corrected",
        #     "balance_element": "NI",
        #     "use_surrogate": True,
        # },
        # "corrected_allSolute": {
        #     "interface_update": "1999_lee_allSolute_corrected",
        #     "balance_element": None,
        #     "use_surrogate": True,
        # },
    }
    actual_config = case.get("actual_database_models", {})
    if actual_config.get("enabled", False):
        for name, spec in actual_config.get("model_specs", {}).items():
            model_specs[name] = {
                "interface_update": spec["interface_update"],
                "balance_element": spec["balance_element"],
                "use_surrogate": False,
            }
    models = {}
    failures = {}
    for name, spec in model_specs.items():
        try:
            model = _build_model(
                case,
                interface_update=spec["interface_update"],
                balance_element=spec["balance_element"],
                use_surrogate=spec.get("use_surrogate", True),
            )
            model.solve(case["solve_time"], iterator=explicitEulerIterator, vIt=case["vIt"], verbose=True, minDtFrac=1e-10)
            models[name] = model
        except Exception as exc:
            failures[name] = str(exc)
            raise exc
    if len(models) == 0:
        failure_text = "; ".join(f"{name}: {message}" for name, message in failures.items())
        raise RuntimeError(f"No model variant completed successfully. Failures: {failure_text}")
    return models, failures


def inventory_residuals(model):
    """Returns the component-wise inventory drift relative to the initial state."""
    final_inventory = np.asarray(model.getTotalInventory(), dtype=np.float64)
    initial_inventory = np.asarray(model._initialInventory, dtype=np.float64)
    return final_inventory - initial_inventory


def plot_results(case, models):
    """Plots interface motion and final composition profiles for each model."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = {
        "corrected_CR": "tab:blue",
        "corrected_NI": "tab:orange",
        "corrected_my_CR": "tab:green",
        "corrected_my_NI": "tab:red",
        "actual_corrected_CR": "tab:purple",
        "actual_corrected_NI": "tab:brown",
        "actual_my_corrected_CR": "tab:pink",
        "corrected_allSolute": "tab:pink",
    }
    linestyle = {
        "corrected_CR": "solid",
        "corrected_NI": "solid",
        "corrected_my_CR": "dotted",
        "corrected_my_NI": "dotted",
        "actual_corrected_CR": "dashed",
        "actual_corrected_NI": "dashed",
        "actual_my_corrected_CR": "dashed",
        "corrected_allSolute": "solid",
    }


    for name, model in models.items():
        t = np.asarray(model.interfaceData._time[: model.interfaceData.N + 1], dtype=np.float64)
        s = np.asarray(model.interfaceData._y[: model.interfaceData.N + 1], dtype=np.float64)
        color = colors.get(name, None)
        ls = linestyle.get(name, None)
        axes[0, 0].plot(t, s, label=name, color=color, linestyle=ls)

        drift = inventory_residuals(model)
        axes[0, 1].plot(["CR", "NI"], drift, "o-", label=name, color=color)

        plot1D(model, elements=["CR"], time=model.currentTime, ax=axes[1, 0], color=color, label=name)
        plot1D(model, elements=["NI"], time=model.currentTime, ax=axes[1, 1], color=color, label=name)

    axes[0, 0].axhline(case["interface_position"], color="0.5", linestyle="--", linewidth=1)
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Interface position")
    axes[0, 0].legend()

    axes[0, 1].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    axes[0, 1].set_ylabel("Final inventory - initial inventory")
    axes[0, 1].legend()

    axes[1, 0].set_title("Final CR profiles")
    axes[1, 1].set_title("Final NI profiles")

    plt.tight_layout()
    if plt.get_backend().lower() != "agg":
        plt.show()


def plot_normalized_left_phase_thickness(models):
    """
    Plots normalized left-phase thickness versus time for all successful runs.

    For this 1D moving-boundary setup, the left-phase thickness is the current
    interface position measured from the left domain boundary, so the
    normalized thickness is ``s(t) / s(0)``.
    """
    nrows_input=3
    fig, axes = plt.subplots(figsize=(8, 5*nrows_input), ncols=1, nrows=nrows_input, sharex=True)
    ax = axes[0]
    ax.xaxis.set_tick_params(labelbottom=True)
    ax2 = axes[1]
    ax2.xaxis.set_tick_params(labelbottom=True)
    ax3 = axes[2]
    colors = {
        "corrected_CR": "tab:blue",
        "corrected_NI": "tab:orange",
        "corrected_my_CR": "tab:green",
        "corrected_my_NI": "tab:red",
        "actual_corrected_CR": "tab:purple",
        "actual_corrected_NI": "tab:brown",
        "actual_my_corrected_CR": "tab:pink",
        "corrected_allSolute": "tab:pink",
    }
    linestyle = {
        "corrected_CR": "solid",
        "corrected_NI": "solid",
        "corrected_my_CR": "dotted",
        "corrected_my_NI": "dotted",
        "actual_corrected_CR": "dashed",
        "actual_corrected_NI": "dashed",
        "actual_my_corrected_CR": "dashed",
        "corrected_allSolute": "solid",
    }
    
    import pandas as pd
    fig9_upperCurve_df = pd.read_csv(EXAMPLES_DIR.joinpath(r'leeAndOh1996_data\fig9_upperCurve_Ni.csv'), names=['t', 'normalized_thickness'])
    fig9_lowerCurve_df = pd.read_csv(EXAMPLES_DIR.joinpath(r'leeAndOh1996_data\fig9_lowerCurve_Cr.csv'), names=['t', 'normalized_thickness'])
    fig9_upperCurve_df = fig9_upperCurve_df.sort_values(by=['t'])
    fig9_lowerCurve_df = fig9_lowerCurve_df.sort_values(by=['t'])
    fig9_upperCurve_df.plot(x='t', y='normalized_thickness', ax=ax, label='Lee & Oh Fig. 9 upper curve (Ni)', color='k')
    fig9_lowerCurve_df.plot(x='t', y='normalized_thickness', ax=ax, label='Lee & Oh Fig. 9 lower curve (Cr)', color='dimgray')
    
    for name, model in models.items():
        t = np.asarray(model.interfaceData._time[: model.interfaceData.N + 1], dtype=np.float64)/3600
        s = np.asarray(model.interfaceData._y[: model.interfaceData.N + 1], dtype=np.float64)
        normalized_left_thickness = s / s[0]
        ax.plot(t, normalized_left_thickness, label=name, color=colors.get(name, None), linestyle=linestyle.get(name, None))

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Normalized left-phase thickness")
    ax.set_title("Left-Phase Thickness vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    ax.set_xlim([1e-4, 10**(int(np.ceil(np.log10(t[-1]))))]); ax.set_xscale("log")
    
    ax_twin = ax.twinx()
    ax_twin.set_ylabel('p', color='red')
    ax_twin.set_ylim([0,1])
    from kawin.diffusion.mesh.MovingBoundaryFD1D import get_moving_boundary_fd_geometry
    for name, model in models.items():
        t = np.asarray(model.interfaceData._time[: model.interfaceData.N + 1], dtype=np.float64)/3600
        s = np.asarray(model.interfaceData._y[: model.interfaceData.N + 1], dtype=np.float64)
        p = [get_moving_boundary_fd_geometry(model.mesh, s_el, model.pstar).p for s_el in s]
        ax_twin.plot(t, p, label=name+" p", color=colors.get(name, None), linestyle=linestyle.get(name, None), alpha=0.5)
    ax_twin.hlines(model.pstar, t[0], t[-1], color='red', linestyle='--', alpha=0.5)
    if model.ignoredNodeRule=="lee_oh_1996_three_region":
        ax_twin.hlines(1-model.pstar, t[0], t[-1], color='red', linestyle='--', alpha=0.5)
        

    # fig2, ax2 = plt.subplots(figsize=(8, 5))
    # ax2.set_xlim([1e-4, 10**(int(np.ceil(np.log10(t[-1]))))]); ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)
    ax2_twin = ax2.twinx()
    ax2.set_ylabel('Component 0', color='blue')
    ax2_twin.set_ylabel('Component 1', color='red')
    for name, model in models.items(): ## could add tqdm progress bar here
        t = np.asarray(model.interfaceData._time, dtype=np.float64)
        t_sub_indices = np.concatenate([np.arange(len(t))[1:100], np.arange(len(t))[100:][::1000]])
        t_sub = t[t_sub_indices].copy()
        
        preOrPost_str = model.fluxGradientMode.split("_")[0]

        def getIntegralAtTime(t_input):
            t_index = np.abs(model.data._time-t_input).argmin()
            timeOfInterest = model.data._time[t_index]
            timeOfInterestPrev = model.data._time[t_index-1]
            int_el0 = model._integrateComponentInventory(
                composition=model.data.y(time=timeOfInterest),
                interface_position=model.interfaceData.y(time=timeOfInterest),
                interface_compositions=(model._interfaceCompositionHistory[f'{preOrPost_str}_left'].y(time=timeOfInterest), model._interfaceCompositionHistory[f'{preOrPost_str}_right'].y(time=timeOfInterest)),
                component_index=0,
                s_for_interp="new",
                s_old=model.interfaceData.y(time=timeOfInterestPrev),
                p_old=get_moving_boundary_fd_geometry(model.mesh, model.interfaceData.y(time=timeOfInterestPrev), model.pstar).p,
                s_new=model.interfaceData.y(time=timeOfInterest),
            )
            int_el1 = model._integrateComponentInventory(
                composition=model.data.y(time=timeOfInterest),
                interface_position=model.interfaceData.y(time=timeOfInterest),
                interface_compositions=(model._interfaceCompositionHistory[f'{preOrPost_str}_left'].y(time=timeOfInterest), model._interfaceCompositionHistory[f'{preOrPost_str}_right'].y(time=timeOfInterest)),
                component_index=1,
                s_for_interp="new",
                s_old=model.interfaceData.y(time=timeOfInterestPrev),
                p_old=get_moving_boundary_fd_geometry(model.mesh, model.interfaceData.y(time=timeOfInterestPrev), model.pstar).p,
                s_new=model.interfaceData.y(time=timeOfInterest),
            )

            return int_el0/model.mesh.zlim[0][-1], int_el1/model.mesh.zlim[0][-1]
        intValsAttSub = np.array([getIntegralAtTime(t_input) for t_input in tqdm.tqdm(t_sub, total=len(t_sub))])
        ax2.hlines(model._initialInventory[0]/model.mesh.zlim[0][-1], t_sub.min()/3600, t_sub.max()/3600, color='blue', linestyle='solid', alpha=0.5)
        ax2_twin.hlines(model._initialInventory[1]/model.mesh.zlim[0][-1], t_sub.min()/3600, t_sub.max()/3600, color='red', linestyle='solid', alpha=0.5)

        ax2.plot(t_sub/3600, intValsAttSub[:, 0], marker='o', label=name+" (Component 0)", color='blue', linestyle='solid')
        ax2_twin.plot(t_sub/3600, intValsAttSub[:, 1], marker='o', label=name+" (Component 1)", color='red', linestyle='solid')
        print(f"{model._initialInventory/model.mesh.zlim[0][-1]}: initialMass")
        print(f"{np.array([0.229945, 0.09035])}: LeeAndOh Table 1 Mass")
        print(f"{(((model._initialInventory/model.mesh.zlim[0][-1])-np.array([0.229945, 0.09035]))/np.array([0.229945, 0.09035])) * 100}: Relative % Error")



    ax2.hlines(0.229945, t_sub.min()/3600, t_sub.max()/3600, color='blue', linestyle='dashed', alpha=0.25)
    ax2_twin.hlines(0.09035, t_sub.min()/3600, t_sub.max()/3600, color='red', linestyle='dashed', alpha=0.25)

    ax3.grid(True, alpha=0.3)
    for name, model in models.items():
        t = np.asarray(model._interfaceCompositionHistory[f'{preOrPost_str}_left']._time, dtype=np.float64)
        interfaceComps = np.asarray(model._interfaceCompositionHistory[f'{preOrPost_str}_left']._y, dtype=np.float64)
        interfaceCompChange = np.linalg.norm(np.diff(interfaceComps, axis=0), axis=1)
        ax3.plot(t[1:]/3600, interfaceCompChange, marker='o', label=name, color=colors.get(name, None), linestyle=linestyle.get(name, None))

    
    plt.tight_layout()
    if plt.get_backend().lower() != "agg":
        plt.show()

    print(f"pstar_input: {pstar_input}")
    print(f"N: {case['nodes']}")

    modelAttributesToPrint = ["integrationMode", "initialInventoryMode", "ignoredNodeRule", "denom_type", "ignoredNodeReconstructionMode", "fluxGradientMode", "multicomponentInterfaceStateUpdate"]
    print([f"{attrStr}: {getattr(model, attrStr)}" for attrStr in modelAttributesToPrint])
    
    return fig, ax


def print_summary(case, models):
    """Prints a compact run summary that is useful when tuning the surrogate."""
    print("Approximate ternary moving-boundary test case")
    print(f"temperature = {case['temperature']} K")
    print(f"domain length = {case['length']}")
    print(f"nodes = {case['nodes']}")
    print(f"initial interface position = {case['interface_position']}")
    print(f"left bulk [CR, NI] = {case['left_bulk']}")
    print(f"right bulk [CR, NI] = {case['right_bulk']}")
    print(f"probe left [CR, NI] = {case.get('probe_left', case['left_bulk'])}")
    print(f"probe right [CR, NI] = {case.get('probe_right', case['right_bulk'])}")
    print("left tie-line endpoints:")
    print(f"  start = {case['left_tieline_start']}")
    print(f"  end   = {case['left_tieline_end']}")
    print("right tie-line endpoints:")
    print(f"  start = {case['right_tieline_start']}")
    print(f"  end   = {case['right_tieline_end']}")
    print("diffusivity matrices:")
    for phase, matrix in case["diffusivities"].items():
        print(f"  {phase} =\n{matrix}")

    if "actual_database_models" in case:
        print("\nActual database model settings")
        print(f"  enabled = {case['actual_database_models'].get('enabled', False)}")
        if case["actual_database_models"].get("enabled", False):
            for name, spec in case["actual_database_models"].get("model_specs", {}).items():
                print(
                    "    "
                    f"{name}: interface_update={spec['interface_update']}, "
                    f"balance_element={spec['balance_element']}"
                )

    if "database_approximation" in case:
        print("\nDatabase approximation settings")
        print(f"  enabled = {case['database_approximation'].get('enabled', False)}")
        print(f"  tdb_path = {case['database_approximation'].get('tdb_path')}")
        print(f"  surrogate_cache_enabled = {case['database_approximation'].get('surrogate_cache_enabled', True)}")
        print(f"  surrogate_cache_policy = {case['database_approximation'].get('surrogate_cache_policy', 'load_or_build')}")
        if "database_surrogate_cache_key" in case:
            print(f"  surrogate_cache_key = {case['database_surrogate_cache_key']}")
        if "database_surrogate_cache_hit" in case:
            print(f"  surrogate_cache_hit = {case['database_surrogate_cache_hit']}")
        if "database_surrogate_cache_paths" in case:
            print(f"  surrogate_cache_npz = {case['database_surrogate_cache_paths']['npz']}")
            print(f"  surrogate_cache_json = {case['database_surrogate_cache_paths']['json']}")
        print(f"  diffusivity_mode = {case.get('database_diffusivity_mode', case['database_approximation'].get('diffusivity_mode', 'single_tieline'))}")
        if case.get('database_diffusivity_mode', case['database_approximation'].get('diffusivity_mode', 'single_tieline')) == "nearest_sample":
            print("  diffusivity inference backend = hybrid (single: SSD, batch: sklearn KNN k=1)")
        if "database_diffusivity_global" in case:
            print(f"  diffusivity global [CR, NI] = {case['database_diffusivity_global']}")
        if "database_diffusivity_surrogate_load_time_s" in case:
            print(
                "  diffusivity surrogate load time (s) = "
                f"{case['database_diffusivity_surrogate_load_time_s']:.3f}"
            )
        if "database_diffusivity_surrogate_build_time_s" in case:
            print(
                "  diffusivity surrogate build time (s) = "
                f"{case['database_diffusivity_surrogate_build_time_s']:.3f}"
            )
        if "database_diffusivity_tieline" in case:
            diff_tie = case["database_diffusivity_tieline"]
            print(f"  sampled left tie-line [CR, NI] = {diff_tie['left_ind']}")
            print(f"  sampled right tie-line [CR, NI] = {diff_tie['right_ind']}")
        if "database_diffusivity_sampling" in case:
            sampling = case["database_diffusivity_sampling"]
            print(f"  diffusivity tieline lambdas = {sampling['tieline_lambdas']}")
            print(f"  diffusivity bulk bbox = {sampling['bulk_bbox']}")
            print(f"  diffusivity bulk spacing = {sampling['bulk_spacing']}")
            print(f"  diffusivity samples by phase = {sampling['sample_count_by_phase']}")
            if "sample_count_by_phase_interface" in sampling:
                print(f"  diffusivity interface-only samples by phase = {sampling['sample_count_by_phase_interface']}")
        if "database_sampled_tielines" in case:
            print("  sampled tie-lines:")
            for sample in case["database_sampled_tielines"]:
                print(
                    "    "
                    f"lambda={sample['lambda']:.3f}, "
                    f"global={sample['global']}, "
                    f"left={sample['left_ind']}, "
                    f"right={sample['right_ind']}"
                )

    print("\nModel results")
    for name, model in models.items():
        left_int, right_int = model.getInterfaceCompositions()
        print(f"{name}:")
        print(f"  final interface position = {model.getInterfacePosition():.8f}")
        print(f"  interface compositions left  = {np.asarray(left_int)}")
        print(f"  interface compositions right = {np.asarray(right_int)}")
        inventoryRes = inventory_residuals(model)
        print(f"  inventory residuals [CR, NI] = {inventoryRes}")
        print(f"  inventory normalized residuals [CR, NI] = {inventoryRes / model.mesh.zlim[0][1]}")


def print_failures(failures):
    """Prints model variants that did not complete."""
    if not failures:
        return
    print("\nModel variants that did not complete")
    for name, message in failures.items():
        print(f"{name}: {message}")


#%%

def main():
    case = build_ternary_case()
    models, failures = solve_models(case)
    print_summary(case, models)
    print_failures(failures)
    # plot_results(case, models)
    # plot_normalized_left_phase_thickness(models)
    # plot_ternary_thermodynamics(case)


if __name__ == "__main__":
    main()
    raise
# raise
#%%
# %config InlineBackend.figure_format = 'svg'
case = build_ternary_case()
models, failures = solve_models(case)
print_summary(case, models)
print_failures(failures)
# plot_results(case, models)
# plot_normalized_left_phase_thickness(models)
# plot_ternary_thermodynamics(case)

 #%%
# %config InlineBackend.figure_format = 'svg'
modelToUse = models["corrected_CR"] 
fig, ax = modelToUse.plotTernaryState()

import pandas as pd
FCC_extractedPts_df = pd.read_csv("leeAndOh1996_data\\FCClineAt1373FromWPDcsv.csv", names=['W(NI)', 'W(CR)'])
BCC_extractedPts_df = pd.read_csv("leeAndOh1996_data\\BCClineAt1373FromWPDcsv.csv", names=['W(NI)', 'W(CR)'])
FCC_extractedPts_df = FCC_extractedPts_df/100
BCC_extractedPts_df = BCC_extractedPts_df/100

def convertNiCrWtToMoleFracs(df_input):
    df = pd.DataFrame.from_dict(df_input.apply(lambda row: v.get_mole_fractions({v.W('CR'): row['W(CR)'], v.W('NI'): row['W(NI)']}, dependent_species='FE', pure_element_mass_dict=modelToUse.therm.db_forPlotting), axis=1).to_list(), orient='columns')
    df.columns = df.columns.astype(str)
    df_input[['X_NI', 'X_CR']] = df[['X_NI', 'X_CR']].copy()
    return df_input

FCC_extractedPts_df = convertNiCrWtToMoleFracs(FCC_extractedPts_df)
BCC_extractedPts_df = convertNiCrWtToMoleFracs(BCC_extractedPts_df)

ax.plot(FCC_extractedPts_df['X_CR'], FCC_extractedPts_df['X_NI'], 'o', fillstyle='none', label='FCC extracted points')
ax.plot(BCC_extractedPts_df['X_CR'], BCC_extractedPts_df['X_NI'], 'o', fillstyle='none', label='BCC extracted points')

# list(models.values())[0].therm
# list(models.values())[0]["database_sampled_tielines"]
# FCC_sampledTielinePts_arr = np.array([tl['phase_ind']['FCC_A1'] for tl in case["database_sampled_tielines"]])
# BCC_sampledTielinePts_arr = np.array([tl['phase_ind']['BCC_A2'] for tl in case["database_sampled_tielines"]])
# ax.plot(FCC_sampledTielinePts_arr[:,0], FCC_sampledTielinePts_arr[:, 1], 'x', markersize=1, fillstyle='none', label='FCC sampled tieline')
# ax.plot(BCC_sampledTielinePts_arr[:,0], BCC_sampledTielinePts_arr[:, 1], 'x', markersize=1, fillstyle='none', label='BCC sampled tieline')


# ax.plot(modelToUse._interfaceCompositionHistory['pre_right']._y[:,0], modelToUse._interfaceCompositionHistory['pre_right']._y[:, 1], 'x', markersize=1, fillstyle='none', label='left interfacial comps')
# ax.plot(modelToUse._interfaceCompositionHistory['pre_left']._y[:,0], modelToUse._interfaceCompositionHistory['pre_left']._y[:, 1], 'x', markersize=1, fillstyle='none', label='left interfacial comps')

globalComps = np.array([[0.23, 0.0904], [0.28476, 0.06109]])
globalComps_evaled = [models["corrected_CR"].therm._tieline_from_lambda(models["corrected_CR"].therm._lambda_from_probe(globalComp)) for globalComp in globalComps]
globalComps_evaled_BCC = np.array([globalComp_evaled['BCC_A2'] for globalComp_evaled in globalComps_evaled])
globalComps_evaled_FCC = np.array([globalComp_evaled['FCC_A1'] for globalComp_evaled in globalComps_evaled])
ax.plot(globalComps[:,0], globalComps[:, 1], 'x', markersize=5, fillstyle='none', label='global comps')
ax.plot(globalComps_evaled_BCC[:,0], globalComps_evaled_BCC[:, 1], 'x', markersize=5, fillstyle='none', label='global comps evaled BCC')
ax.plot(globalComps_evaled_FCC[:,0], globalComps_evaled_FCC[:, 1], 'x', markersize=5, fillstyle='none', label='global comps evaled FCC')
ax.plot(np.array([models["corrected_CR"].therm.left_probe, models["corrected_CR"].therm.right_probe])[:,0], np.array([models["corrected_CR"].therm.left_probe, models["corrected_CR"].therm.right_probe])[:, 1], color='k', label='probeLine')

ax.legend()
# ax.set_xlim([0.1, 0.5])
# ax.set_ylim([0, 0.2])
ax.set_xlim([0.15, 0.35])
ax.set_ylim([0.04, 0.125])

# %%
import pandas as pd
fig9_upperCurve_df = pd.read_csv(EXAMPLES_DIR.joinpath(r'leeAndOh1996_data\fig9_upperCurve_Ni.csv'), names=['t', 'normalized_thickness'])
fig9_lowerCurve_df = pd.read_csv(EXAMPLES_DIR.joinpath(r'leeAndOh1996_data\fig9_lowerCurve_Cr.csv'), names=['t', 'normalized_thickness'])
fig9_upperCurve_df = fig9_upperCurve_df.sort_values(by=['t'])
fig9_lowerCurve_df = fig9_lowerCurve_df.sort_values(by=['t'])
 

# %%


from scipy import optimize

def getTielineOfGlobalComposition(globalComp, T):
    def dotWithTlPerp(lam):
        tlPt1, tlPt2 = models["corrected_CR"].therm.getInterfacialComposition(lam, T, returnMeta=False, xIsLambda=True)
        tlDir = tlPt1-tlPt2
        tlPerpDir = np.array([-tlDir[1], tlDir[0]])
        return np.dot(globalComp - tlPt1, tlPerpDir)

    sol = optimize.root_scalar(
                dotWithTlPerp,
                # bracket=[0, 1],
                bracket=[0.0, 1.0],
                method="brentq",
                # rtol=1e-14,
                xtol=1e-10,
                maxiter=100,
            )
    return models["corrected_CR"].therm.getInterfacialComposition(sol.root, T, returnMeta=True, xIsLambda=True)
t0 = time.perf_counter()
for i in range(1000):
    getTielineOfGlobalComposition(globalComp=np.array([0.23, 0.0904]), T=1373)
t1 = time.perf_counter()
print(t1-t0)
getTielineOfGlobalComposition(globalComp=np.array([0.1235, 0.0001]), T=1373)
# %%
