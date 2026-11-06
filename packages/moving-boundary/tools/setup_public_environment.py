"""Create an isolated offline stock-kawin environment with bounded subprocesses."""
from pathlib import Path
import subprocess
import sys
import hashlib
import zipfile
import json

project = Path(__file__).resolve().parents[1]
base = project.parents[2]
env = base / 'mb-public-050'
assert env.resolve() != Path(sys.prefix).resolve(), 'Run setup from the activated shared kawin environment, not its target clone'
wheel = project / '.verification/kawin-0.5.0-py3-none-any.whl'
if not wheel.exists():
    raise FileNotFoundError('Restore the cached unmodified kawin 0.5.0 wheel under .verification before offline setup')
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == '9f7a1497c5dff4314536860ac6fd0b96a68243dd681db8eb106dbe33fe25cbaf'
def run(args, timeout=600):
    """Use inherited activated DLL paths and bound all environment operations."""
    subprocess.run(args, check=True, timeout=timeout, stdin=subprocess.DEVNULL)
if not (env / 'python.exe').exists():
    run(['C:/ProgramData/anaconda3/Scripts/conda.exe', 'create', '--offline', '--clone',
         sys.prefix, '--prefix', str(env), '-y'])
run([str(env / 'python.exe'), '-m', 'pip', 'install', str(wheel), '--no-deps', '--no-index', '--force-reinstall'])
site = env / 'Lib/site-packages'
with zipfile.ZipFile(wheel) as archive:
    files = [n for n in archive.namelist() if n.startswith('kawin/') and not n.endswith('/')]
    for name in files:
        assert (site / name).read_bytes() == archive.read(name), name
run([str(env / 'python.exe'), '-I', '-c',
     'import kawin,numpy as np; from importlib.metadata import version; '
     'assert version("kawin")=="0.5.0"; print(kawin.__file__); '
     'np.testing.assert_allclose(np.linalg.solve([[2.,1.],[1.,3.]],[1.,2.]),[.2,.6])'], 60)
(project / '.verification/public-environment.json').write_text(json.dumps(dict(
    python=str(env/'python.exe'), host=str(site), version='0.5.0',
    stock_files_verified=len(files), wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest()),indent=2))
print('Stock wheel files verified:',len(files),flush=True)
