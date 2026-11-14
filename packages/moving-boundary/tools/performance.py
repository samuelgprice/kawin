"""Manual package performance runs and explicit, advisory baseline comparisons.

Run through the repository's activated kawin command wrapper. Every run builds a
fresh private install, then starts an isolated worker in the selected public
Conda host. Nothing is installed into that host. No acceptance module is used.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from performance_results import compare, format_comparison

PROJECT = Path(__file__).resolve().parents[1]
TOOLS = PROJECT / "tools"
OUTPUT = PROJECT / ".verification/benchmarks"
ACTIVATE = Path("C:/ProgramData/anaconda3/Scripts/activate.bat")


def digest(path):
    """Hash an artifact without using Git or importing the package."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_identity():
    """Record optional revision/dirty state; exports and missing Git remain usable."""
    try:
        # Explicit git-dir also works for this Windows worktree's ownership layout.
        root = PROJECT.parent.parent
        marker = root / ".git"
        gitdir = marker
        if marker.is_file():
            gitdir = (root / marker.read_text(encoding="utf-8").strip().removeprefix("gitdir: ")).resolve()
        if not gitdir.exists():
            return dict(available=False, reason="No Git metadata (export)")
        base = ["git", f"--git-dir={gitdir}", f"--work-tree={root}"]
        def read(*args):
            return subprocess.run([*base, *args], capture_output=True, text=True,
                                  check=True, timeout=10, stdin=subprocess.DEVNULL).stdout.strip()
        return dict(available=True, revision=read("rev-parse", "HEAD"),
                    branch=read("branch", "--show-current"), dirty=read("status", "--porcelain", "--untracked-files=normal"))
    except (OSError, subprocess.SubprocessError) as exc:
        return dict(available=False, reason=str(exc))


def new_folder(root):
    """Reserve a unique run/comparison directory, refusing all reuse."""
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = root / name
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def run_command(args, *, cwd, timeout, log, env=None):
    """Bound a noninteractive subprocess; retain its full log and fail visibly."""
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != "nt")
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            else:
                import signal
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
            raise
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}); see {log}\n" + log.read_text(encoding="utf-8")[-6000:])


def run_suite(args):
    """Build/install privately and launch a fresh worker with a suite deadline."""
    started = time.perf_counter()
    folder = new_folder(args.output.resolve())
    print(f"Run folder: {folder}", flush=True)
    config_path = TOOLS / "performance_cases.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    profile = config["profiles"][args.mode]
    cases = [dict(case, steps=profile["steps"], workload_version=config["version"],
                  workload_sha256=digest(TOOLS / "performance_cases.py")) for case in config["cases"]
             if not args.case or case["id"] in args.case]
    if not cases or (args.case and set(args.case)-{case["id"] for case in cases}):
        raise ValueError("Unknown or empty case selection")
    if args.environment:
        prefix = args.environment.resolve()
    else:
        pointer = PROJECT / ".verification/public-environment.json"
        if pointer.exists():
            prefix = Path(json.loads(pointer.read_text(encoding="utf-8"))["python"]).parent
        else:
            prefix = Path(sys.prefix)
    host = prefix / ("Lib/site-packages" if os.name == "nt" else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    python = prefix / ("python.exe" if os.name == "nt" else "bin/python")
    if not host.is_dir() or not python.is_file():
        raise ValueError(f"Invalid environment: {prefix}")
    wheels, target = folder / "wheels", folder / "target"
    temp = folder / "tmp"
    temp.mkdir()
    build_env = dict(os.environ, TMP=str(temp), TEMP=str(temp), TMPDIR=str(temp))
    # Build through activated kawin, not a host with an editable checkout hook.
    run_command([sys.executable, "-m", "pip", "wheel", str(PROJECT), "--no-deps", "--no-build-isolation",
                 "--no-index", "--wheel-dir", str(wheels)], cwd=folder, timeout=120, log=folder / "build.log", env=build_env)
    wheel, = wheels.glob("*.whl")
    target.mkdir(exist_ok=False)
    run_command([sys.executable, "-m", "pip", "install", str(wheel), "--no-deps", "--no-index",
                 "--target", str(target)], cwd=folder, timeout=120, log=folder / "install.log", env=build_env)
    identity = dict(git=git_identity(), wheel_sha256=digest(wheel),
                    config_sha256=digest(config_path), package_files={
                        str(path.relative_to(target)).replace("\\", "/"): digest(path)
                        for path in sorted((target / "moving_boundary").rglob("*.py"))},
                    target=str(target), host=str(host), executable=str(python),
                    created_utc=datetime.now(timezone.utc).isoformat())
    request = dict(target=str(target), host=str(host), identity=identity, cases=cases,
                   warmups=profile["warmups"], samples=args.samples or profile["samples"],
                   measurement=dict(protocol=config["protocol"], timer="perf_counter",
                                    warmups=profile["warmups"],
                                    harness_sha256=digest(TOOLS / "performance_worker.py"),
                                    caches="warm process/JIT; fresh analytic providers and model each sample",
                                    setup="first setup nested inside solve; excludes construction",
                                    solve="complete fresh public solve, includes first setup, finalization and counting wrappers",
                                    finalize="first postSolve nested within solve",
                                    reset="completed validated model.reset; excludes re-setup"))
    request_path = folder / "request.json"
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    # -S avoids editable .pth/sitecustomize hooks. Explicit site directories supply
    # distributions without executing their hooks or exposing the research root.
    bootstrap = folder / "worker.py"
    bootstrap.write_text("import runpy, sys\n" +
        f"sys.path[:0] = {[str(target), str(host), str(TOOLS)]!r}\n" +
        f"sys.argv = [{str(TOOLS / 'performance_worker.py')!r}, {str(request_path)!r}]\n" +
        f"runpy.run_path({str(TOOLS / 'performance_worker.py')!r}, run_name='__main__')\n", encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    env.update(PYTHONDONTWRITEBYTECODE="1", NUMBA_CACHE_DIR=str(folder / "numba-cache"))
    command = [str(python), "-S", str(bootstrap)]
    if os.name == "nt":
        # Activation configures native DLL lookup; never invoke Conda Python bare.
        batch = folder / "worker.cmd"
        batch.write_text(f'@echo off\ncall "{ACTIVATE}" "{prefix}"\nif errorlevel 1 exit /b 1\n'
                         + subprocess.list2cmdline(command) + "\n", encoding="utf-8")
        command = f'"{os.environ.get("COMSPEC", "cmd.exe")}" /d /s /c ""{batch}""'
    try:
        run_command(command, cwd=folder, env=env, timeout=args.timeout, log=folder / "worker.log")
    finally:
        if (folder / "worker.log").exists():
            print((folder / "worker.log").read_text(encoding="utf-8"), flush=True)
    elapsed = time.perf_counter()-started
    (folder / "wall-time.json").write_text(json.dumps(dict(total_seconds=elapsed)), encoding="utf-8")
    print(f"Total including private build/install: {elapsed:.2f} s", flush=True)


def main(argv=None):
    """Dispatch independent run/compare actions; never select or advance a baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run", help="create a new unique local measurement")
    run.add_argument("--mode", choices=("smoke", "standard"), default="standard")
    run.add_argument("--environment", type=Path, help="public kawin Conda prefix; default local pointer, otherwise active environment")
    run.add_argument("--output", type=Path, default=OUTPUT)
    run.add_argument("--case", action="append", help="explicit case ID; repeat for a subset")
    run.add_argument("--samples", type=int, help="override repetitions; retained in results")
    run.add_argument("--timeout", type=int, default=1200, help="worker deadline in seconds (default 20 minutes)")
    diff = commands.add_parser("compare", help="compare two explicitly chosen results.json files")
    diff.add_argument("baseline", type=Path)
    diff.add_argument("candidate", type=Path)
    diff.add_argument("--slowdown", type=float, default=.20)
    diff.add_argument("--noise-multiplier", type=float, default=6.)
    diff.add_argument("--minimum-seconds", type=float, default=.002)
    diff.add_argument("--output", type=Path, default=OUTPUT / "comparisons")
    args = parser.parse_args(argv)
    if args.action == "run":
        if (args.samples is not None and args.samples < 1) or args.timeout < 1:
            parser.error("samples and timeout must be positive")
        run_suite(args)
    else:
        result = compare(json.loads(args.baseline.read_text(encoding="utf-8")),
                         json.loads(args.candidate.read_text(encoding="utf-8")),
                         slowdown=args.slowdown, noise_multiplier=args.noise_multiplier,
                         minimum_seconds=args.minimum_seconds)
        result.update(baseline=str(args.baseline.resolve()), candidate=str(args.candidate.resolve()))
        text = format_comparison(result)
        folder = new_folder(args.output.resolve())
        (folder / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (folder / "comparison.txt").write_text(text+"\n", encoding="utf-8")
        print(text + f"\nSaved: {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
