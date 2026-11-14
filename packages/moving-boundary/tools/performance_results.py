"""Pure timing summaries and advisory comparison; no simulation or approval hooks."""
import math
import statistics
import json

SCHEMA = 1


def summarize(samples):
    """Retain raw positive seconds and summarize typical time and robust spread."""
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("Timing samples must be finite, positive and nonempty")
    median = statistics.median(samples)
    mad = statistics.median(abs(x-median) for x in samples)
    return dict(samples=list(samples), median=median, min=min(samples), max=max(samples),
                mad=mad, relative_mad=mad/median)


def differences(a, b, prefix=""):
    """List differing nested metadata fields, including absent fields."""
    result = []
    for key in sorted(a.keys() | b.keys()):
        path = f"{prefix}.{key}" if prefix else key
        if key not in a or key not in b:
            result.append(path)
        elif isinstance(a[key], dict) and isinstance(b[key], dict):
            result.extend(differences(a[key], b[key], path))
        elif a[key] != b[key]:
            result.append(path)
    return result


def compare(baseline, candidate, *, slowdown=.20, noise_multiplier=6., minimum_seconds=.002):
    """Compare only compatible case/metric pairs; flags request human review.

    Ratio = candidate median / baseline median. Review requires a relative
    increase beyond both the configured fraction and six relative MADs, and
    an absolute increase above 2 ms by default. This is not a statistical test.
    Environment differences retain ratios but label them confounded. Work
    changes are reported independently, including counts for individual runs.
    """
    if any(not math.isfinite(x) or x < 0 for x in (slowdown, noise_multiplier, minimum_seconds)):
        raise ValueError("Review thresholds must be finite and nonnegative")
    if baseline.get("schema") != SCHEMA or candidate.get("schema") != SCHEMA:
        raise ValueError("Unsupported result schema")
    env_diff = differences(baseline["environment"], candidate["environment"])
    protocol_diff = differences(baseline["measurement"], candidate["measurement"])
    result = dict(environment_differences=env_diff, protocol_differences=protocol_diff,
                  thresholds=dict(slowdown=slowdown, noise_multiplier=noise_multiplier,
                                  minimum_seconds=minimum_seconds), cases=[])
    result["run_status"] = dict(baseline=baseline.get("status", "unknown"), candidate=candidate.get("status", "unknown"))
    if "identity" in baseline and "identity" in candidate:
        result["package_content_changed"] = baseline["identity"]["package_files"] != candidate["identity"]["package_files"]
        result["revisions"] = dict(baseline=baseline["identity"]["git"].get("revision"),
                                   candidate=candidate["identity"]["git"].get("revision"))
    old, new = baseline["cases"], candidate["cases"]
    for name in sorted(old.keys() | new.keys()):
        row = dict(case=name, metrics={})
        result["cases"].append(row)
        if name not in old or name not in new:
            row["status"] = "missing_baseline" if name not in old else "missing_candidate"
            continue
        a, b = old[name], new[name]
        row["configuration_differences"] = differences(a["configuration"], b["configuration"])
        if (a.get("status") != "ok" or b.get("status") != "ok"
                or baseline.get("status") != "ok" or candidate.get("status") != "ok"):
            row["status"] = "invalid_run"
            continue
        if row["configuration_differences"] or protocol_diff:
            row["status"] = "incompatible"
            continue
        row["status"] = "environment_differs" if env_diff else "comparable"
        row["work_changed"] = ({json.dumps(item, sort_keys=True) for item in a["work"]}
                               != {json.dumps(item, sort_keys=True) for item in b["work"]})
        row["baseline_work"], row["candidate_work"] = a["work"], b["work"]
        for metric in sorted(a["metrics"].keys() | b["metrics"].keys()):
            if metric not in a["metrics"] or metric not in b["metrics"]:
                row["metrics"][metric] = dict(status="missing_metric")
                continue
            # Recompute summaries from retained data, rather than trusting cached medians.
            x, y = summarize(a["metrics"][metric]["samples"]), summarize(b["metrics"][metric]["samples"])
            ratio = y["median"]/x["median"]
            guard = max(slowdown, noise_multiplier*max(x["relative_mad"], y["relative_mad"]))
            flag = ratio > 1+guard and y["median"]-x["median"] > minimum_seconds
            row["metrics"][metric] = dict(status="review" if flag else "observed", ratio=ratio,
                                          baseline_seconds=x["median"], candidate_seconds=y["median"],
                                          relative_noise_guard=guard)
    return result


def format_comparison(result):
    """Render concise per-case ratios, warnings and review flags."""
    lines = ["Ratios are candidate/baseline; REVIEW is advisory (no timing pass/fail)."]
    if "package_content_changed" in result:
        lines.append("Installed package content: " + ("changed" if result["package_content_changed"] else "identical (repeat measurement; no historical regression evidence)"))
    lines.append("Run status: " + ", ".join(f"{key}={value}" for key, value in result["run_status"].items()))
    for group in ("environment_differences", "protocol_differences"):
        if result[group]:
            lines.append(group + ": " + ", ".join(result[group]))
    for row in result["cases"]:
        metrics = " ".join(f"{name}={value['ratio']:.3f}x" + (" REVIEW" if value["status"] == "review" else "")
                           if "ratio" in value else f"{name}=missing"
                           for name, value in row["metrics"].items())
        lines.append(f"{row['case']}: {row['status']} {metrics}" + (" WORK CHANGED" if row.get("work_changed") else ""))
        if row.get("configuration_differences"):
            lines.append("  configuration: " + ", ".join(row["configuration_differences"]))
    return "\n".join(lines)
