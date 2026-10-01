"""
Generators for non-uniform transformed (Landau) grids used by the Illingworth
front-fixing moving-boundary solvers.

The planar Illingworth models solve each phase on a fixed grid in a
transformed coordinate on ``[0, 1]``:

- left phase:   ``z = s * u``              (interface at ``u = 1``)
- right phase:  ``z = s + (R - s) * v``    (interface at ``v = 0``)
- middle phase (three-phase model): ``z = s_ab + (s_bc - s_ab) * w``
  (interfaces at ``w = 0`` and ``w = 1``)

The returned arrays are passed directly as ``transformed_u_grid`` /
``transformed_v_grid`` (``MovingBoundaryIllingworthTernaryFD1DModel``,
``MovingBoundaryIllingworthFD1DModel``) or ``transformed_grids``
(``MovingBoundaryIllingworthTernaryThreePhaseFD1DModel``).

Numerical notes
---------------
- Grids are fixed in transformed space, so the physical node spacing scales
  with the current phase width (``s``, ``R - s``, ...) as the interface moves.
- The two-phase solvers evaluate the interface flux from the
  interface-adjacent transformed intervals (``1 - u[-2]`` and ``v[1]``).
  Clustering nodes at the interface therefore refines the gradient that sets
  the interface velocity.
- The bulk assembly is a three-point non-uniform finite-difference /
  finite-volume stencil. Its accuracy degrades where neighbouring intervals
  differ strongly in size, so prefer smooth stretchings (``tanh``) or mild
  power-law exponents, and check ``describe_landau_grid(...)["max_adjacent_ratio"]``.

Every generator returns a float64 array with at least three nodes, strictly
increasing, with endpoints exactly ``0.0`` and ``1.0``.
"""

import numpy as np

_ENDPOINT_ATOL = 1e-14
_DEFAULT_MIN_INTERVAL = 1e-12
# Relative spread below which interval widths are treated as uniform when plotting.
_UNIFORM_SPACING_RTOL = 1e-9


def _validate_node_count(n_nodes):
    """Returns ``n_nodes`` as an int, requiring at least three nodes."""
    if isinstance(n_nodes, bool) or int(n_nodes) != n_nodes:
        raise ValueError(f"n_nodes must be an integer, got {n_nodes!r}.")
    n = int(n_nodes)
    if n < 3:
        raise ValueError(f"n_nodes must be at least 3, got {n}.")
    return n


def _finalize(values):
    """Snaps endpoints to exactly 0 and 1 and returns a float64 copy."""
    grid = np.asarray(values, dtype=np.float64).copy()
    grid[0] = 0.0
    grid[-1] = 1.0
    return grid


def uniform_grid(n_nodes):
    """Returns ``n_nodes`` equally spaced points on ``[0, 1]``."""
    return _finalize(np.linspace(0.0, 1.0, _validate_node_count(n_nodes)))


def power_law_grid(n_nodes, exponent, cluster_at=0.0):
    """
    Returns a power-law stretched grid on ``[0, 1]``.

    With ``x = linspace(0, 1, n_nodes)`` the grid is ``x**exponent`` when
    ``cluster_at=0.0`` and its mirror image ``1 - (1 - x)**exponent`` when
    ``cluster_at=1.0``. ``exponent > 1`` clusters nodes at ``cluster_at``,
    ``exponent == 1`` is uniform, and ``0 < exponent < 1`` clusters nodes at
    the opposite end.

    The interval at the clustered end is ``(1 / (n_nodes - 1))**exponent``, so
    large exponents quickly produce very small cells beside much larger ones.
    Power-law grids have no two-sided form; use ``tanh_grid(..., "both")``
    to cluster at both ends.
    """
    n = _validate_node_count(n_nodes)
    p = float(exponent)
    if not np.isfinite(p) or p <= 0.0:
        raise ValueError(f"exponent must be a positive finite number, got {exponent!r}.")
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    if isinstance(cluster_at, str):
        raise ValueError(
            f"power_law_grid supports cluster_at=0.0 or 1.0, got {cluster_at!r}. "
            "Use tanh_grid or linear_spacing_grid with cluster_at='both' to cluster at both ends."
        )
    if cluster_at == 0.0:
        return _finalize(x**p)
    if cluster_at == 1.0:
        return _finalize(1.0 - (1.0 - x) ** p)
    raise ValueError(f"cluster_at must be 0.0 or 1.0, got {cluster_at!r}.")


def tanh_grid(n_nodes, beta, cluster_at=0.0):
    """
    Returns a hyperbolic-tangent stretched grid on ``[0, 1]``.

    With ``x = linspace(0, 1, n_nodes)``:

    - ``cluster_at=0.0``:    ``1 + tanh(beta * (x - 1)) / tanh(beta)``
    - ``cluster_at=1.0``:    ``tanh(beta * x) / tanh(beta)``
    - ``cluster_at="both"``: ``0.5 * (1 + tanh(beta * (2x - 1)) / tanh(beta))``

    ``beta >= 0`` sets the clustering strength; ``beta == 0`` returns the
    uniform grid exactly. The spacing varies smoothly, so neighbouring
    intervals never jump abruptly in size. The ``"both"`` form suits the
    middle phase of the three-phase model, where both ends are interfaces.
    """
    n = _validate_node_count(n_nodes)
    b = float(beta)
    if not np.isfinite(b) or b < 0.0:
        raise ValueError(f"beta must be a non-negative finite number, got {beta!r}.")
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    if isinstance(cluster_at, str):
        if cluster_at != "both":
            raise ValueError(f"cluster_at must be 0.0, 1.0, or 'both', got {cluster_at!r}.")
        if b == 0.0:
            return _finalize(x)
        return _finalize(0.5 * (1.0 + np.tanh(b * (2.0 * x - 1.0)) / np.tanh(b)))
    if cluster_at not in (0.0, 1.0):
        raise ValueError(f"cluster_at must be 0.0, 1.0, or 'both', got {cluster_at!r}.")
    if b == 0.0:
        return _finalize(x)
    if cluster_at == 0.0:
        return _finalize(1.0 + np.tanh(b * (x - 1.0)) / np.tanh(b))
    return _finalize(np.tanh(b * x) / np.tanh(b))


def linear_spacing_grid(n_nodes, spacing_ratio, cluster_at=0.0):
    """
    Returns a grid whose interval widths vary linearly from one end.

    With ``N = n_nodes - 1`` intervals and uniform width ``h = 1 / N``, the
    interval at ``cluster_at`` has width ``spacing_ratio * h`` and widths
    change by a constant increment away from it so that they sum to one:

    - ``cluster_at=0.0`` or ``1.0``: widths run linearly from
      ``spacing_ratio * h`` to ``(2 - spacing_ratio) * h`` at the other end.
      Requires ``0 < spacing_ratio < 2``.
    - ``cluster_at="both"``: widths are ``spacing_ratio * h`` at both ends and
      vary linearly to a peak (or trough) at the centre, with the centre
      width set by the unit-sum constraint. ``spacing_ratio`` must keep every
      width positive, and at least 4 nodes are needed.

    ``spacing_ratio < 1`` refines the chosen end(s), ``1`` is uniform and
    ``> 1`` coarsens them. Node positions are quadratic in the node index,
    so the end-to-end width ratio is at most ``(2 - r) / r``. Because the
    increment is constant in absolute size, the largest neighbouring-width
    ratio sits at the fine end and is ``1 + 2(1 - r) / (r (N - 1))`` for the
    one-sided form; small ``spacing_ratio`` with few nodes therefore gives a
    sharp size jump next to the finest interval.
    """
    n = _validate_node_count(n_nodes)
    r = float(spacing_ratio)
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError(f"spacing_ratio must be a positive finite number, got {spacing_ratio!r}.")
    n_intervals = n - 1
    k = np.arange(n_intervals, dtype=np.float64) / (n_intervals - 1)
    if isinstance(cluster_at, str):
        if cluster_at != "both":
            raise ValueError(f"cluster_at must be 0.0, 1.0, or 'both', got {cluster_at!r}.")
        if n_intervals < 3 and r != 1.0:
            raise ValueError("linear_spacing_grid with cluster_at='both' needs at least 4 nodes, since both end intervals are fixed.")
        t = 1.0 - np.abs(2.0 * k - 1.0)
    elif cluster_at == 0.0:
        t = k
    elif cluster_at == 1.0:
        t = 1.0 - k
    else:
        raise ValueError(f"cluster_at must be 0.0, 1.0, or 'both', got {cluster_at!r}.")
    if r == 1.0:
        return uniform_grid(n)
    h = 1.0 / n_intervals
    # widths = a + b * t with t = 0 at the chosen end(s); b enforces sum(widths) = 1.
    a = r * h
    b = (1.0 - n_intervals * a) / float(np.sum(t))
    widths = a + b * t
    if not np.all(widths > 0.0):
        raise ValueError(
            f"spacing_ratio={r:g} makes some interval widths non-positive for "
            f"n_nodes={n} and cluster_at={cluster_at!r}; use a smaller ratio "
            "(one-sided grids require spacing_ratio < 2)."
        )
    return _finalize(np.concatenate(([0.0], np.cumsum(widths))))


def _uniform_method(n_nodes, cluster_at=None, **_stretching_kwargs):
    """
    Registry adapter for uniform grids.

    Ignores the clustering location and any stretching parameters, so a
    per-phase ``{"method": "uniform"}`` override can coexist with shared
    parameters such as ``beta`` in ``two_phase_landau_grids``.
    """
    return uniform_grid(n_nodes)


# Registry of named methods for landau_grid. Each entry is called as
# ``func(n_nodes, cluster_at=..., **method_kwargs)``.
_METHODS = {
    "uniform": _uniform_method,
    "power": power_law_grid,
    "tanh": tanh_grid,
    "linear": linear_spacing_grid,
}

# Maps (phase, cluster) to the transformed-coordinate end(s) to refine.
_CLUSTER_LOCATIONS = {
    ("left", "interface"): 1.0,
    ("left", "far"): 0.0,
    ("right", "interface"): 0.0,
    ("right", "far"): 1.0,
    ("middle", "interface"): "both",
}

_PHASES = ("left", "right", "middle")


def validate_landau_grid(grid, name="grid", min_interval=_DEFAULT_MIN_INTERVAL):
    """
    Validates a transformed grid and returns a float64 copy.

    Applies the same rules as the solvers' ``_validate_transformed_grid``
    (1D, at least three finite nodes, endpoints within 1e-14 of 0 and 1,
    strictly increasing) and additionally rejects grids whose smallest
    interval is below ``min_interval`` (pass ``0`` or ``None`` to disable).
    Endpoints are snapped to exactly 0 and 1.
    """
    values = np.asarray(grid, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if len(values) < 3:
        raise ValueError(f"{name} must contain at least three nodes.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.isclose(values[0], 0.0, rtol=0.0, atol=_ENDPOINT_ATOL) or not np.isclose(values[-1], 1.0, rtol=0.0, atol=_ENDPOINT_ATOL):
        raise ValueError(f"{name} must start at 0 and end at 1.")
    values = _finalize(values)
    intervals = np.diff(values)
    if not np.all(intervals > 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    if min_interval is not None and min_interval > 0.0 and float(np.min(intervals)) < float(min_interval):
        raise ValueError(
            f"{name} has a smallest transformed interval of {float(np.min(intervals)):.3e}, "
            f"below min_interval={float(min_interval):.3e}. Weaken the stretching "
            "(smaller beta / exponent closer to 1) or reduce the node count."
        )
    return values


def landau_grid(n_nodes, method="uniform", *, phase="right", cluster="interface", min_interval=_DEFAULT_MIN_INTERVAL, **method_kwargs):
    """
    Builds a transformed grid for one phase, oriented by physical meaning.

    Parameters
    ----------
    n_nodes : int
        Number of grid nodes (at least 3).
    method : {"uniform", "power", "tanh", "linear"}
        Stretching family. ``"power"`` requires ``exponent=``; ``"tanh"``
        requires ``beta=``; ``"linear"`` requires ``spacing_ratio=`` (width
        of the interval at the clustered end relative to uniform spacing).
    phase : {"left", "right", "middle"}
        Which phase the grid is for. This fixes where the interface lies in
        transformed space: ``u = 1`` (left), ``v = 0`` (right), or both ends
        (middle phase of the three-phase model).
    cluster : {"interface", "far"}
        Whether to refine toward the interface(s) or toward the fixed far
        boundary. Ignored for ``method="uniform"``. ``"far"`` is not defined
        for the middle phase, and the middle phase needs ``method="tanh"``
        or ``method="linear"``.
    min_interval : float or None
        Smallest allowed transformed interval; see ``validate_landau_grid``.
    **method_kwargs
        Parameters forwarded to the chosen generator.
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown grid method {method!r}; expected one of {sorted(_METHODS)}.")
    if phase not in _PHASES:
        raise ValueError(f"phase must be one of {_PHASES}, got {phase!r}.")
    if method == "uniform":
        cluster_at = None
    else:
        if (phase, cluster) not in _CLUSTER_LOCATIONS:
            raise ValueError(f"cluster={cluster!r} is not supported for phase={phase!r}.")
        cluster_at = _CLUSTER_LOCATIONS[(phase, cluster)]
    grid = _METHODS[method](n_nodes, cluster_at=cluster_at, **method_kwargs)
    return validate_landau_grid(grid, name=f"{phase} {method} grid", min_interval=min_interval)


def two_phase_landau_grids(n_left, n_right, method="uniform", cluster="interface", *, left_kwargs=None, right_kwargs=None, min_interval=_DEFAULT_MIN_INTERVAL, **shared_kwargs):
    """
    Returns ``(u_grid, v_grid)`` for the two-phase Illingworth solvers.

    ``method``, ``cluster`` and ``shared_kwargs`` (e.g. ``beta=2.0``) apply to
    both phases. ``left_kwargs`` / ``right_kwargs`` override any of these per
    phase, including ``method`` and ``cluster``, e.g.
    ``right_kwargs={"method": "uniform"}``.
    """
    grids = []
    for n_nodes, phase, overrides in ((n_left, "left", left_kwargs), (n_right, "right", right_kwargs)):
        options = {"method": method, "cluster": cluster, **shared_kwargs, **(overrides or {})}
        grids.append(landau_grid(n_nodes, phase=phase, min_interval=min_interval, **options))
    return tuple(grids)


def three_phase_landau_grids(n_a, n_b, n_c, method="uniform", cluster="interface", *, a_kwargs=None, b_kwargs=None, c_kwargs=None, min_interval=_DEFAULT_MIN_INTERVAL, **shared_kwargs):
    """
    Returns ``(grid_a, grid_b, grid_c)`` for ``transformed_grids=`` in the
    three-phase Illingworth solver.

    Phase A is treated as a left phase, C as a right phase and B as the
    middle phase (interfaces at both ends). Non-uniform middle-phase grids
    require ``method="tanh"`` or ``"linear"`` with ``cluster="interface"``; use ``b_kwargs``
    to choose a different method or parameters for B than for A and C.
    """
    grids = []
    for n_nodes, phase, overrides in ((n_a, "left", a_kwargs), (n_b, "middle", b_kwargs), (n_c, "right", c_kwargs)):
        options = {"method": method, "cluster": cluster, **shared_kwargs, **(overrides or {})}
        grids.append(landau_grid(n_nodes, phase=phase, min_interval=min_interval, **options))
    return tuple(grids)


def describe_landau_grid(grid, phase=None):
    """
    Summarizes transformed-grid spacing quality.

    Returns a dict with ``n_nodes``, ``start_interval``, ``end_interval``,
    ``min_interval``, ``max_interval`` and ``max_adjacent_ratio`` (the largest
    size ratio between neighbouring intervals, >= 1). When ``phase`` is
    ``"left"`` or ``"right"``, ``interface_interval`` and ``far_interval``
    are added; for ``"middle"``, ``interface_intervals`` holds both ends.
    """
    values = np.asarray(grid, dtype=np.float64).reshape(-1)
    intervals = np.diff(values)
    ratios = intervals[1:] / intervals[:-1]
    summary = {
        "n_nodes": int(len(values)),
        "start_interval": float(intervals[0]),
        "end_interval": float(intervals[-1]),
        "min_interval": float(np.min(intervals)),
        "max_interval": float(np.max(intervals)),
        "max_adjacent_ratio": float(np.max(np.maximum(ratios, 1.0 / ratios))) if len(ratios) else 1.0,
    }
    if phase == "left":
        summary["interface_interval"] = summary["end_interval"]
        summary["far_interval"] = summary["start_interval"]
    elif phase == "right":
        summary["interface_interval"] = summary["start_interval"]
        summary["far_interval"] = summary["end_interval"]
    elif phase == "middle":
        summary["interface_intervals"] = (summary["start_interval"], summary["end_interval"])
    elif phase is not None:
        raise ValueError(f"phase must be one of {_PHASES} or None, got {phase!r}.")
    return summary


def summarize_landau_grid(grid, phase=None, name=None, physical_length=None, plot=False, bins="auto", cdf_log=False):
    """
    Prints a spacing summary of a transformed grid and optionally plots it.

    The printout reports the minimum and maximum interval (with their
    interval indices), their ratio, the largest neighbouring-interval ratio
    and, when ``phase`` is given, the interface- and far-boundary intervals.
    Spacings are in transformed units on ``[0, 1]``; if ``physical_length``
    is given (e.g. ``s`` for a left phase or ``R - s`` for a right phase),
    the min/max spacings are also printed scaled by it. Because the grid is
    fixed in transformed space, physical spacings change as the phase width
    changes during a run.

    With ``plot=True`` a three-panel figure is created: the grid nodes on a
    line (interface location(s) marked when ``phase`` is given), a
    histogram of the interval widths, and the empirical cumulative
    distribution (CDF) of the interval widths. The CDF spacing axis is
    linear by default; ``cdf_log=True`` makes it logarithmic, which spreads
    out strongly clustered grids whose widths span decades.
    Matplotlib is imported only then. If all widths agree to within a
    relative 1e-9 (a uniform grid up to round-off), the histogram uses fixed
    bins spanning +/-10% of the mean width, instead of automatic bins that would collapse onto the round-off spread;
    the CDF axis uses the same +/-10% range. Both spacing panels mark the
    uniform reference spacing ``1 / (n_nodes - 1)`` with a dashed vertical
    line. Since the widths always sum to one, this reference equals the mean
    width and therefore always lies between the smallest and largest width.

    Returns
    -------
    (summary, fig)
        ``summary`` is the ``describe_landau_grid`` dict plus
        ``min_interval_index``, ``max_interval_index`` and
        ``uniform_interval`` (``1 / (n_nodes - 1)``); ``fig`` is the
        matplotlib figure, or ``None`` when ``plot=False``.
    """
    values = np.asarray(grid, dtype=np.float64).reshape(-1)
    summary = describe_landau_grid(values, phase=phase)
    intervals = np.diff(values)
    summary["min_interval_index"] = int(np.argmin(intervals))
    summary["max_interval_index"] = int(np.argmax(intervals))
    summary["uniform_interval"] = 1.0 / len(intervals)

    label = name if name is not None else (f"{phase} phase grid" if phase is not None else "grid")
    lines = [
        f"Landau grid summary: {label} ({summary['n_nodes']} nodes, {len(intervals)} intervals)",
        f"  min spacing        : {summary['min_interval']:.6e}  (interval {summary['min_interval_index']})",
        f"  max spacing        : {summary['max_interval']:.6e}  (interval {summary['max_interval_index']})",
        f"  max/min spacing    : {summary['max_interval'] / summary['min_interval']:.4g}",
        f"  max adjacent ratio : {summary['max_adjacent_ratio']:.4g}",
        f"  uniform reference  : {summary['uniform_interval']:.6e}  (1 / {len(intervals)} intervals)",
    ]
    if "interface_interval" in summary:
        lines.append(f"  interface spacing  : {summary['interface_interval']:.6e}")
        lines.append(f"  far spacing        : {summary['far_interval']:.6e}")
    elif "interface_intervals" in summary:
        lines.append(f"  interface spacings : {summary['interface_intervals'][0]:.6e}, {summary['interface_intervals'][1]:.6e}")
    if physical_length is not None:
        length = float(physical_length)
        lines.append(f"  physical min/max   : {summary['min_interval'] * length:.6e} / {summary['max_interval'] * length:.6e}  (length {length:.6e})")
    print("\n".join(lines))

    fig = None
    if plot:
        import matplotlib.pyplot as plt

        fig, (ax_nodes, ax_hist, ax_cdf) = plt.subplots(3, 1, figsize=(7.0, 7.0), gridspec_kw={"height_ratios": [1, 2, 2]})
        ax_nodes.axhline(0.0, color="0.6", linewidth=1.0, zorder=0)
        ax_nodes.plot(values, np.zeros_like(values), linestyle="none", marker="|", markersize=16, color="C0")
        interface_locations = {"left": (1.0,), "right": (0.0,), "middle": (0.0, 1.0)}.get(phase, ())
        for i, location in enumerate(interface_locations):
            ax_nodes.axvline(location, color="C3", linestyle="--", linewidth=1.0, label="interface" if i == 0 else None)
        if interface_locations:
            ax_nodes.legend(loc="upper center", frameon=False, fontsize="small")
        ax_nodes.set_xlim(-0.02, 1.02)
        ax_nodes.set_yticks([])
        ax_nodes.set_xlabel("transformed coordinate")
        ax_nodes.set_title(label)

        mean_interval = float(np.mean(intervals))
        is_uniform = float(np.ptp(intervals)) <= _UNIFORM_SPACING_RTOL * mean_interval
        if is_uniform:
            # Uniform up to round-off: automatic bins would span only ~1e-16
            # and produce an empty-looking plot with an offset axis.
            ax_hist.hist(intervals, bins=21, range=(0.9 * mean_interval, 1.1 * mean_interval), color="C0", edgecolor="white")
        else:
            ax_hist.hist(intervals, bins=bins, color="C0", edgecolor="white")
        uniform_label = f"uniform spacing 1/{len(intervals)} = {summary['uniform_interval']:.4e}"
        ax_hist.axvline(summary["uniform_interval"], color="k", linestyle="--", linewidth=1.2, label=uniform_label)
        ax_hist.legend(loc="upper right", frameon=False, fontsize="small")
        ax_hist.ticklabel_format(axis="x", useOffset=False)
        ax_hist.set_xlabel("transformed spacing")
        ax_hist.set_ylabel("number of intervals")

        # Empirical CDF: fraction of intervals with width <= x, as a step function.
        sorted_intervals = np.sort(intervals)
        fractions = np.arange(1, len(sorted_intervals) + 1) / len(sorted_intervals)
        ax_cdf.step(
            np.concatenate(([sorted_intervals[0]], sorted_intervals)),
            np.concatenate(([0.0], fractions)),
            where="post",
            color="C0",
        )
        ax_cdf.plot(sorted_intervals, fractions, linestyle="none", marker="o", markersize=3, color="C0")
        ax_cdf.axvline(summary["uniform_interval"], color="k", linestyle="--", linewidth=1.2, label=uniform_label)
        ax_cdf.legend(loc="lower right", frameon=False, fontsize="small")
        if is_uniform:
            x_low, x_high = 0.9 * mean_interval, 1.1 * mean_interval
        elif cdf_log:
            x_low, x_high = sorted_intervals[0] / 1.15, sorted_intervals[-1] * 1.15
        else:
            pad = 0.05 * (sorted_intervals[-1] - sorted_intervals[0])
            x_low, x_high = sorted_intervals[0] - pad, sorted_intervals[-1] + pad
        if cdf_log:
            ax_cdf.set_xscale("log")
            ax_cdf.set_xlim(x_low, x_high)
            # Default log ticks label only powers of ten, which leaves narrow
            # ranges (common for these grids) with one or overlapping labels.
            from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

            if x_high / x_low < 10.0:
                ax_cdf.xaxis.set_major_locator(FixedLocator(np.geomspace(x_low, x_high, 5)))
            else:
                ax_cdf.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            ax_cdf.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.3g}"))
            ax_cdf.xaxis.set_minor_formatter(NullFormatter())
            ax_cdf.set_xlabel("transformed spacing (log scale)")
        else:
            ax_cdf.set_xlim(x_low, x_high)
            ax_cdf.ticklabel_format(axis="x", useOffset=False)
            ax_cdf.set_xlabel("transformed spacing")
        ax_cdf.set_ylim(0.0, 1.05)
        ax_cdf.grid(True, which="both", alpha=0.3)
        ax_cdf.set_ylabel("cumulative fraction")
        fig.tight_layout()
    return summary, fig
