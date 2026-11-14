# Saved performance baselines

These are manually selected performance checkpoints for periodic comparisons.
They do not run in ordinary tests or acceptance, change scientific references,
or automatically advance after a successful measurement. Keep existing records;
add a new dated directory when choosing another milestone.

## Initial checkpoint: 2026-11-13 Windows standard

`2026-11-13-windows-standard/results.json` preserves the nine-case standard run
`20261114T023030Z-eeb7a395`, measured on November 13, 2026 at 20:30 CST
(November 14 at 02:30 UTC). Each case has one warm-up and seven measured
samples, with 1,600 scheduled steps per trajectory. All cases passed their
benchmark validity checks. Worker suite time was 374.56 seconds; total elapsed
time including the private build/install was 390.75 seconds (about 6 min 31 s).

The recorded Git revision is `6beb34193a7b6ea817da3113df334d86bb62c2fe`.
The tree was dirty: the benchmark tooling was not yet committed and the user's
example settings were modified. Do not relabel this as a measurement made at
`5b6542b` or a later revision. The installed per-file fingerprints, wheel hash,
workload hash and harness hash identify what was actually measured. Full
configurations, all timing samples, summaries, work counts and numerical
validation summaries are retained in the JSON.

The host used CPython 3.11.15, public kawin 0.5.0, NumPy 2.4.3 and SciPy 1.17.1
on Windows/AMD64. Detailed compiler, BLAS, CPU and thread settings are preserved.
These are synthetic/public workloads; no private database or Thermo-Calc
measurements are included. Raw logs, installations and caches remain local
under `.verification/benchmarks/`.

## Compare a later milestone

See [the performance guide](../../docs/PERFORMANCE.md) for running a new standard
measurement. Use the same machine, environment and workload where possible.
Export the new result with `export_result.py` before comparing it with this
baseline. The export replaces local paths with labels and hostnames with stable
machine IDs, and retains the Git dirty flag without its filename listing. It
does not change samples or compatibility settings. Applying the same export to
both results avoids path-redaction differences being mistaken for environment
changes. Use this export policy for future saved checkpoints too.

From the extraction root, substitute the new run's path for `NEW_RUN`:

```powershell
cmd.exe /d /c 'call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3\envs\kawin" && python packages/moving-boundary/benchmarks/baselines/export_result.py "NEW_RUN/results.json" "NEW_RUN/portable-results.json"'
cmd.exe /d /c 'call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3\envs\kawin" && python packages/moving-boundary/tools/performance.py compare packages/moving-boundary/benchmarks/baselines/2026-11-13-windows-standard/results.json "NEW_RUN/portable-results.json"'
```

The export refuses an existing output file. The original result is unchanged;
its SHA256 is recorded for local provenance, along with total installation/run
time when available. The comparator accepts the exported schema unchanged.
Version/workload differences and review flags still apply. Repeat suspected
slowdowns under matched conditions; these observations are not causal proof of
a regression. Smoke runs have a different workload and are not comparable to
this standard baseline.

When recording another reviewed checkpoint, commit its exported JSON and a
short explanation of the measured revision and conditions. JSON may require
`git add -f` because of local ignore rules. Keep a stable anchor as well as the
previous benchmarked milestone, so gradual slowdowns remain visible.
