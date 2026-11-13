"""Strict layout-correction rules: an additional structural proof, never a relaxation (LEDGERS R09c).

A layout correction is a reviewed change to how a history is *stored* while every computed value
stays the same. It is checked against each case's previously selected expected artifacts
(``acceptance.expected_reference``), on top of every normal correctness gate, and never with a
numerical tolerance: a numerical change, even one within public tolerance, fails the proof.

Rule ``r09-data-tail`` (R09c removes the legacy duplicate final ``data`` row of binary and
two-phase ternary models at recording interval 1). For one case, ``classify`` compares the
reference arrays with the candidate arrays and returns one of three verdicts:

- ``identical``: the same key set, and every array has the same dtype, shape and bytes;
- ``tail-removal``: the same key set; every array outside the declared pairs is identical; and
  every declared pair that changed lost exactly its last row, which was a bitwise duplicate of
  the row before it, in both its time and its values array (``candidate == reference[:-1]``);
- ``other``: anything else, with the reasons in ``problems``.

Declared pairs, recognized from the key set only: ``data/time`` + ``data/values`` (run cases),
``model<i>/data/time`` + ``model<i>/data/values`` (contracts), and, for LEGACY only,
``<mode>__data_time`` + ``<mode>__data_y``. The only normalization is the caller's existing
``normalize(case, key, array)`` (O06's documented metadata cache path); declared pair arrays are
never normalized. Verdict records are structural only (key names and verdicts, no values or
deviations), so they can be committed for private cases too.
"""
import re

import numpy as np

R09_DATA_TAIL = "r09-data-tail"
RULES = (R09_DATA_TAIL,)
_RUN_OR_CONTRACT = re.compile(r"^((?:model\d+/)?)data/time$")
_LEGACY = re.compile(r"^(.+)__data_time$")


def _identity(case, key, value):
    return value


def bit_identical(left, right):
    """Same shape, dtype and bytes (signed zeros and NaN payloads included)."""
    left, right = np.asarray(left), np.asarray(right)
    return left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes()


def declared_pairs(case, keys):
    """The ``(time key, values key)`` pairs a ``r09-data-tail`` correction may shorten in ``case``."""
    keys = set(keys)
    pairs = []
    for key in sorted(keys):
        match = _RUN_OR_CONTRACT.match(key)
        if match and match.group(1) + "data/values" in keys:
            pairs.append((key, match.group(1) + "data/values"))
            continue
        match = _LEGACY.match(key)
        if case == "LEGACY" and match and match.group(1) + "__data_y" in keys:
            pairs.append((key, match.group(1) + "__data_y"))
    return pairs


def _tail_problems(time_key, values_key, old_time, new_time, old_values, new_values):
    """Why one changed pair is not an exact one-row duplicate-tail removal (empty when it is)."""
    where = f"{time_key} + {values_key}"
    old_time, new_time = np.asarray(old_time), np.asarray(new_time)
    old_values, new_values = np.asarray(old_values), np.asarray(new_values)
    if old_time.ndim != 1 or old_values.ndim < 1 or old_values.shape[0] != old_time.shape[0]:
        return [where + ": reference pair is not a time series"]
    rows = old_time.shape[0]
    if rows < 2:
        return [where + ": reference pair has no row to remove"]
    if new_time.dtype != old_time.dtype or new_values.dtype != old_values.dtype:
        return [where + ": dtype changed"]
    if new_time.shape != (rows - 1,) or new_values.shape != (rows - 1, *old_values.shape[1:]):
        return [where + ": not exactly one row removed"]
    problems = []
    if not (bit_identical(new_time, old_time[:-1]) and bit_identical(new_values, old_values[:-1])):
        problems.append(where + ": kept rows differ from the reference")
    if old_time[-1].tobytes() != old_time[-2].tobytes() or old_values[-1].tobytes() != old_values[-2].tobytes():
        problems.append(where + ": removed row is not a bitwise duplicate of the row before it")
    return problems


def classify(case, reference, candidate, *, rule=R09_DATA_TAIL, normalize=_identity):
    """Classify ``candidate`` against ``reference`` (mappings of key to array) under ``rule``.

    Returns ``dict(case, rule, verdict, corrected, problems)`` where ``verdict`` is ``identical``,
    ``tail-removal`` or ``other`` (module docstring), ``corrected`` lists the shortened pairs as
    ``[time key, values key]`` and ``problems`` explains an ``other`` verdict.
    """
    assert rule in RULES, "unknown layout correction: " + str(rule)
    reference_keys, candidate_keys = set(reference), set(candidate)
    result = dict(case=case, rule=rule, verdict="other", corrected=[], problems=[])
    if reference_keys != candidate_keys:
        result["problems"].append(f"key set differs: missing {sorted(reference_keys - candidate_keys)}, "
                                  f"extra {sorted(candidate_keys - reference_keys)}")
        return result
    pairs = declared_pairs(case, reference_keys)
    paired = {key for pair in pairs for key in pair}
    changed = set()
    for key in sorted(reference_keys):
        old, new = reference[key], candidate[key]
        if key not in paired:
            old, new = normalize(case, key, old), normalize(case, key, new)
        if not bit_identical(old, new):
            changed.add(key)
    outside = sorted(changed - paired)
    result["problems"].extend("array changed outside the declared pairs: " + key for key in outside)
    for time_key, values_key in pairs:
        if not ({time_key, values_key} & changed):
            continue
        if not ({time_key, values_key} <= changed):
            result["problems"].append(f"{time_key} + {values_key}: only one array of the pair changed")
            continue
        found = _tail_problems(time_key, values_key, reference[time_key], candidate[time_key],
                               reference[values_key], candidate[values_key])
        if found:
            result["problems"].extend(found)
        else:
            result["corrected"].append([time_key, values_key])
    if not result["problems"]:
        result["verdict"] = "tail-removal" if result["corrected"] else "identical"
    return result


def structural(result):
    """The committed form of a verdict: rule, verdict and corrected key pairs (no values)."""
    return dict(rule=result["rule"], verdict=result["verdict"], corrected=[list(pair) for pair in result["corrected"]],
                problems=list(result["problems"]))
