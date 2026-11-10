"""F7: array-wise npz identity, with ZIP container differences reported separately."""
import importlib.util
import zipfile
from pathlib import Path

import numpy as np


def _tool():
    spec = importlib.util.spec_from_file_location(
        "npz_identity_tool", Path(__file__).parents[1] / "tools/acceptance.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


ARRAYS = dict(a=np.array([0.0, 1.5]), b=np.arange(6, dtype=np.int64).reshape(2, 3), c=np.array("legacy"))


def _save(path, arrays, compressed=True):
    (np.savez_compressed if compressed else np.savez)(path, **arrays)
    return path


def test_identical_files_are_array_and_container_identical(tmp_path):
    left, right = _save(tmp_path / "l.npz", ARRAYS), _save(tmp_path / "r.npz", ARRAYS)
    result = _tool().npz_identity(left, right)
    assert result["array_identical"] and result["container"]["byte_identical"]
    assert set(result["arrays"]) == {"a", "b", "c"}


def test_container_difference_is_metadata_only(tmp_path):
    left, right = _save(tmp_path / "l.npz", ARRAYS), _save(tmp_path / "r.npz", ARRAYS, compressed=False)
    result = _tool().npz_identity(left, right)
    assert result["array_identical"]
    assert not result["container"]["byte_identical"] and not result["container"]["members_equal"]
    assert {m["compress_type"] for m in result["container"]["adopted"]} == {zipfile.ZIP_STORED}


def test_signed_zero_dtype_and_key_differences_fail(tmp_path):
    tool = _tool()
    adopted = _save(tmp_path / "adopted.npz", ARRAYS)
    zero = _save(tmp_path / "zero.npz", dict(ARRAYS, a=np.array([-0.0, 1.5])))
    dtype = _save(tmp_path / "dtype.npz", dict(ARRAYS, b=ARRAYS["b"].astype(np.int32)))
    keys = _save(tmp_path / "keys.npz", dict(a=ARRAYS["a"], b=ARRAYS["b"], d=np.zeros(1)))
    for path, key in ((zero, "a"), (dtype, "b")):
        result = tool.npz_identity(path, adopted)
        assert not result["array_identical"] and not result["arrays"][key]["bit_identical"]
    result = tool.npz_identity(keys, adopted)
    assert not result["array_identical"]
    assert result["generated_only"] == ["d"] and result["adopted_only"] == ["c"]
