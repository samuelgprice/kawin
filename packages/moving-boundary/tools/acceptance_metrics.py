"""Deviation measurements with the frozen elementwise tolerance and private-data boundary."""
import numpy as np

PRIVATE_CASES = frozenset({"O01", "O02", "O03a", "O03b", "O04", "O07"})
QUANTITIES = ("interface", "profiles", "eta", "inventory", "accepted_times", "analytic_error")


def array_metrics(reference, actual, *, rtol, atol):
    """Measure finite errors and bytes without changing acceptance decisions.

    Relative error is abs(actual-reference)/abs(reference). A nonzero error at
    zero reference has unbounded relative error (JSON null plus an explicit flag).
    NaNs at matching positions and equal infinities contribute no finite error;
    nonfinite mismatches are unbounded. The tolerance fraction is the maximum
    elementwise abs(error)/(atol+rtol*abs(reference)), not a ratio of two maxima.
    Signed zero/NaN payload changes still make bit_identical false.
    """
    left, right = np.asarray(reference), np.asarray(actual)
    result = dict(bit_identical=False, max_absolute_deviation=None, max_relative_deviation=None,
        relative_deviation_unbounded=False, max_tolerance_fraction=None,
        tolerance_fraction_unbounded=False, worst_error=None, worst_allowed_error=None,
        tolerance=dict(rtol=float(rtol), atol=float(atol), rule="atol + rtol * abs(reference)"))
    if left.shape != right.shape:
        result.update(structural_error="shape mismatch", within_10x_of_tolerance=True)
        return result
    result["bit_identical"] = left.dtype == right.dtype and left.tobytes() == right.tobytes()
    dtype = np.complex128 if left.dtype.kind == "c" or right.dtype.kind == "c" else np.float64
    a, b = left.astype(dtype,copy=False), right.astype(dtype,copy=False)
    finite = np.isfinite(a) & np.isfinite(b)
    special_match = (np.isnan(a) & np.isnan(b)) | (np.isinf(a) & (a == b))
    special_mismatch = bool(np.any(~finite & ~special_match))
    with np.errstate(over="ignore",invalid="ignore"):
        difference = np.abs(b[finite]-a[finite])
    absolute_unbounded = special_mismatch or not bool(np.all(np.isfinite(difference)))
    magnitude = np.abs(a[finite])
    allowance = atol + rtol*magnitude
    zero_reference_error = (magnitude == 0) & (difference != 0)
    relative_unbounded = bool(np.any(zero_reference_error)) or special_mismatch
    with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
        relative = np.divide(difference,magnitude,out=np.zeros_like(difference),where=magnitude!=0)
    relative_unbounded = relative_unbounded or not bool(np.all(np.isfinite(relative)))
    fraction_unbounded = bool(np.any((allowance == 0) & (difference != 0))) or special_mismatch
    with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
        fraction = np.divide(difference,allowance,out=np.zeros_like(difference),where=allowance!=0)
    fraction_unbounded = fraction_unbounded or not bool(np.all(np.isfinite(fraction)))
    worst = int(np.argmax(fraction)) if fraction.size else None
    if fraction_unbounded and difference.size and np.any((allowance == 0) & (difference != 0)):
        worst = int(np.flatnonzero((allowance == 0) & (difference != 0))[0])
    result.update(
        max_absolute_deviation=None if absolute_unbounded else float(np.max(difference,initial=0.)),
        absolute_deviation_unbounded=absolute_unbounded,
        max_relative_deviation=None if relative_unbounded else float(np.max(relative,initial=0.)),
        relative_deviation_unbounded=relative_unbounded,
        zero_reference_nonzero_errors=int(np.count_nonzero(zero_reference_error)),
        max_tolerance_fraction=None if fraction_unbounded else float(np.max(fraction,initial=0.)),
        tolerance_fraction_unbounded=fraction_unbounded,
        within_10x_of_tolerance=fraction_unbounded or bool(np.max(fraction,initial=0.) >= .1),
        worst_error=None if worst is None or not np.isfinite(difference[worst]) else float(difference[worst]),
        worst_allowed_error=None if worst is None else float(allowance[worst]),
        nonfinite_mismatch=special_mismatch)
    return result


def quantity_metrics(arrays):
    """Aggregate an applicable quantity while retaining each array's actual tolerance.

    Analytic-error arrays can have different physical scales, so their per-array
    tolerances remain explicit. Worst normalized error is comparable across them.
    """
    if not arrays:
        return dict(applicable=False,bit_identical=None,max_absolute_deviation=None,
                    max_relative_deviation=None,tolerance=None)
    unbounded = any(item.get("tolerance_fraction_unbounded") or item.get("structural_error") for item in arrays.values())
    worst_key = max(arrays,key=lambda key: (arrays[key].get("tolerance_fraction_unbounded",False),
                                           arrays[key].get("max_tolerance_fraction") or 0.))
    worst = arrays[worst_key]
    relative_unbounded = any(item["relative_deviation_unbounded"] or item.get("structural_error") for item in arrays.values())
    absolute_unbounded = any(item.get("absolute_deviation_unbounded") or item.get("structural_error") for item in arrays.values())
    return dict(applicable=True,bit_identical=all(item["bit_identical"] for item in arrays.values()),
        max_absolute_deviation=None if absolute_unbounded else max(item["max_absolute_deviation"] for item in arrays.values()),
        max_relative_deviation=None if relative_unbounded else max(item["max_relative_deviation"] for item in arrays.values()),
        relative_deviation_unbounded=relative_unbounded,
        max_tolerance_fraction=None if unbounded else worst["max_tolerance_fraction"],
        tolerance_fraction_unbounded=unbounded, worst_array=worst_key,
        worst_error=worst["worst_error"], worst_allowed_error=worst["worst_allowed_error"],
        tolerance={key:item["tolerance"] for key,item in arrays.items()},
        within_10x_of_tolerance=any(item["within_10x_of_tolerance"] for item in arrays.values()), arrays=arrays)


def summarize(cases):
    """Count run identity and find the worst elementwise tolerance utilization."""
    identical = sum(item["bit_identical"] for item in cases.values())
    close = sum(item["passed"] and not item["bit_identical"] for item in cases.values())
    candidates=[]
    near=[]
    for case,record in cases.items():
        for quantity,stats in record["quantities"].items():
            if not stats.get("applicable") or quantity=="stop_status":
                continue
            entry=dict(case=case,quantity=quantity,array=stats["worst_array"],
                fraction=stats["max_tolerance_fraction"],unbounded=stats["tolerance_fraction_unbounded"],
                error=stats["worst_error"],allowance=stats["worst_allowed_error"])
            candidates.append(entry)
            if stats["within_10x_of_tolerance"]:
                near.append(entry)
    worst=max(candidates,key=lambda e:(e["unbounded"],e["fraction"] or 0.)) if candidates else None
    return dict(bit_identical_runs=int(identical),within_tolerance_not_identical_runs=int(close),
        failed_runs=sum(not item["passed"] for item in cases.values()),worst=worst,within_10x=near)


def public_summary(summary):
    """Remove every TC-derived numerical metric while retaining coverage labels."""
    result={key:value for key,value in summary.items() if key not in {"worst","within_10x"}}
    def redact(entry):
        if entry and entry["case"] in PRIVATE_CASES:
            return dict(case=entry["case"],quantity=entry["quantity"],private_metrics=True)
        return entry
    result["worst"]=redact(summary["worst"])
    result["within_10x"]=[redact(entry) for entry in summary["within_10x"]]
    return result


def public_cases(cases):
    """Export quantity summaries; retain array detail and all TC metrics only locally."""
    result={}
    for case,record in cases.items():
        if case in PRIVATE_CASES:
            result[case]=dict(private_metrics=True,reason="TC-derived numerical metrics retained only in ignored output")
            continue
        exported={key:value for key,value in record.items() if key!="quantities"}
        if "quantities" in record:
            exported["quantities"]={}
            for quantity,stats in record["quantities"].items():
                compact={key:value for key,value in stats.items() if key!="arrays"}
                tolerances=stats.get("tolerance")
                if isinstance(tolerances,dict) and tolerances:
                    unique=[]
                    for settings in tolerances.values():
                        if settings not in unique:
                            unique.append(settings)
                    if len(unique)==1:
                        compact["tolerance"]=unique[0]
                exported["quantities"][quantity]=compact
        result[case]=exported
    return result


def print_summary(label, summary):
    """Print a compact deviation summary without exposing private numerical values."""
    safe=public_summary(summary)
    text=(f"{label} bit-identical: {safe['bit_identical_runs']} runs; within tolerance but not identical: "
          f"{safe['within_tolerance_not_identical_runs']} runs; failed: {safe['failed_runs']}")
    worst=safe["worst"]
    if worst:
        if worst.get("private_metrics"):
            text+=f" (worst: {worst['case']}/{worst['quantity']}; private numerical metrics in ignored output)"
        else:
            text+=f" (worst: {worst['case']}/{worst['quantity']}, error {worst['error']} vs allowance {worst['allowance']})"
    print(text,flush=True)
    if safe["within_10x"]:
        print("Within 10x of tolerance: "+", ".join(item['case']+'/'+item['quantity'] for item in safe["within_10x"]),flush=True)
    else:
        print("Within 10x of tolerance: none",flush=True)
