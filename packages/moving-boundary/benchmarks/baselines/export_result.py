"""Export a benchmark result with portable paths, preserving all measurements.

Run this on both saved baselines and future candidates before using the existing
performance comparator. Source/workload fingerprints and protocol fields remain
unchanged; only local paths, the hostname and the verbose Git status are curated.
This utility neither runs a benchmark nor selects or overwrites a baseline.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re


POLICY = "portable-paths-v1"


def portable_result(raw, source_bytes, run_id, total_seconds=None):
    """Copy schema-1 results, replacing local identity details consistently.

    NumPy build paths are normalized on both sides of future comparisons. Version,
    compiler, BLAS, hardware and thread settings remain available for detecting
    confounded comparisons. A stable hostname digest still distinguishes machines.
    All case configurations, samples, summaries, work counts and validation results
    are preserved exactly. The raw source digest permits checking the local original.
    """
    if raw.get("schema") != 1 or "archive" in raw:
        raise ValueError("Expected an original schema-1 benchmark result")
    result = copy.deepcopy(raw)
    identity, environment = result["identity"], result["environment"]
    executable = identity["executable"].replace("\\", "/")
    prefix = executable.rsplit("/", 1)[0]

    def portable_paths(value):
        if isinstance(value, dict):
            return {key: portable_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable_paths(item) for item in value]
        if isinstance(value, str):
            path = value.replace("\\", "/")
            if path == prefix or path.startswith(prefix + "/"):
                return "<PUBLIC_ENV>" + path[len(prefix):]
            if re.match(r"^(?:[A-Za-z]:/|/)", path):
                return "<LOCAL_PATH>/" + path.rstrip("/").rsplit("/", 1)[-1]
        return value

    # This field contains JSON text, rather than a nested object in schema 1.
    numpy_config = json.loads(environment["numpy_config"])
    environment["numpy_config"] = json.dumps(portable_paths(numpy_config), sort_keys=True)
    hostname = environment.get("node")
    if hostname:
        environment["node"] = "machine-" + hashlib.sha256(hostname.encode()).hexdigest()[:16]
    identity["target"] = "<PRIVATE_TARGET>"
    identity["host"] = portable_paths(identity["host"])
    identity["executable"] = portable_paths(identity["executable"])
    if "dirty" in identity["git"]:
        identity["git"]["dirty"] = bool(identity["git"]["dirty"])
    result["archive"] = {
        "policy": POLICY,
        "source_run": run_id,
        "source_result_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "total_seconds_including_install": total_seconds,
        "git_status_note": "Dirty flag retained; unrelated working-tree filenames omitted.",
    }
    return result


def main(argv=None):
    """Export one explicit local result to a new file; never overwrite either input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="original results.json")
    parser.add_argument("output", type=Path, help="new portable result path")
    args = parser.parse_args(argv)
    source_bytes = args.source.read_bytes()
    wall_path = args.source.with_name("wall-time.json")
    total = json.loads(wall_path.read_text(encoding="utf-8"))["total_seconds"] if wall_path.exists() else None
    result = portable_result(json.loads(source_bytes), source_bytes, args.source.parent.name, total)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Exported {len(result['cases'])} cases to {args.output}")


if __name__ == "__main__":
    main()
