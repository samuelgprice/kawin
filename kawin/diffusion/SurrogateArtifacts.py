"""Verified, configuration-aware artifact bundles for diffusion surrogates.

The low-level surrogate ``save`` and ``load`` methods intentionally remain
available for archival round trips.  This module adds the stricter layer used
when a caller wants to reuse an expensive surrogate calculation: every build
input is fingerprinted, every persisted member is checksummed, and a manifest
is published only after all members have been written successfully.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .MovingBoundarySurrogates import TernaryMovingBoundaryThermodynamicsSurrogate


ARTIFACT_SCHEMA_VERSION = 1
SURROGATE_SERIALIZATION_VERSION = 1


class SurrogateArtifactError(RuntimeError):
    """Base class for verified surrogate artifact failures."""


class SurrogateArtifactCompatibilityError(SurrogateArtifactError):
    """The saved artifact was not constructed from the expected build inputs."""


class SurrogateArtifactIntegrityError(SurrogateArtifactError):
    """The artifact manifest or one of its members is missing or corrupted."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def callable_source_sha256(value: Callable[..., Any]) -> str:
    """Hash the source text of a build-recipe callable.

    An unavailable source is a compatibility failure rather than a reason to
    silently weaken the artifact fingerprint.
    """

    try:
        source = inspect.getsource(value)
    except (OSError, TypeError) as exc:
        raise SurrogateArtifactCompatibilityError(
            f"Cannot determine source identity for {value!r}."
        ) from exc
    return _sha256_bytes(source.encode("utf-8"))


def _canonical_array(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("Object arrays cannot be included in a surrogate build specification.")
    if array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and not np.little_endian):
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
    array = np.ascontiguousarray(array)
    return {
        "__ndarray__": True,
        "dtype": array.dtype.str.replace(">", "<"),
        "shape": list(array.shape),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def canonicalize_build_spec(value: Any) -> Any:
    """Convert a build specification to a deterministic JSON-compatible form.

    NumPy arrays are represented by dtype, shape, and byte digest. Dictionary
    keys are sorted during serialization, while sequence order remains
    significant because sample ordering can affect a numerical build.
    Non-finite floating-point values are rejected.
    """

    if isinstance(value, np.ndarray):
        return _canonical_array(value)
    if isinstance(value, np.generic):
        return canonicalize_build_spec(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): canonicalize_build_spec(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonicalize_build_spec(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Surrogate build specifications require finite floating-point values.")
        return value
    raise TypeError(f"Unsupported surrogate build specification value: {type(value).__name__}.")


def surrogate_build_fingerprint(build_spec: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Return the canonical build specification and its deterministic SHA-256."""

    canonical = canonicalize_build_spec(build_spec)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return canonical, _sha256_bytes(encoded)


def _differences(expected: Any, actual: Any, path: str = "build_spec", limit: int = 20) -> list[str]:
    differences: list[str] = []

    def visit(left: Any, right: Any, location: str):
        if len(differences) >= limit:
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                child = f"{location}.{key}"
                if key not in left:
                    differences.append(f"{child}: unexpected saved value {right[key]!r}")
                elif key not in right:
                    differences.append(f"{child}: missing from saved artifact")
                else:
                    visit(left[key], right[key], child)
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                differences.append(f"{location}: expected length {len(left)}, saved length {len(right)}")
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f"{location}[{index}]")
            return
        if left != right:
            differences.append(f"{location}: expected {left!r}, saved {right!r}")

    visit(expected, actual, path)
    return differences


def _safe_member_path(bundle_path: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise SurrogateArtifactIntegrityError(f"Unsafe artifact member path: {relative_path!r}.")
    resolved_bundle = bundle_path.resolve()
    resolved_member = (bundle_path / relative).resolve()
    if resolved_member != resolved_bundle and resolved_bundle not in resolved_member.parents:
        raise SurrogateArtifactIntegrityError(f"Artifact member escapes its bundle: {relative_path!r}.")
    return resolved_member


def _jsonl_schema(path: Path) -> int | None:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                try:
                    header = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SurrogateArtifactIntegrityError(f"Invalid JSONL header in {path.name}: {exc}") from exc
                if not isinstance(header, dict) or header.get("record_type") != "metadata":
                    raise SurrogateArtifactIntegrityError(
                        f"JSONL member {path.name} does not begin with a metadata record."
                    )
                return header.get("schema_version")
    raise SurrogateArtifactIntegrityError(f"JSONL member {path.name} is empty.")


@dataclass(frozen=True)
class SurrogateArtifactBundle:
    """A verified surrogate together with its manifest and bundle location."""

    path: Path
    surrogate: TernaryMovingBoundaryThermodynamicsSurrogate
    manifest: dict[str, Any]

    def member_path(self, role: str) -> Path:
        """Return the verified path for a named artifact member."""

        try:
            relative = self.manifest["members"][str(role)]["path"]
        except KeyError as exc:
            raise SurrogateArtifactIntegrityError(f"Artifact has no member with role {role!r}.") from exc
        return _safe_member_path(self.path, relative)

    @classmethod
    def load_published(cls, bundle_path: str | Path) -> "SurrogateArtifactBundle":
        """Load and verify a bundle against its own published build specification.

        This inspection-oriented loader permits comparison of artifacts with
        different build fingerprints. It still performs every integrity and
        internal-consistency check used by :meth:`load`; it simply does not
        claim that either artifact matches the caller's current build settings.
        Standalone NPZ archives are not accepted.
        """

        bundle_path = Path(bundle_path)
        manifest_path = bundle_path / "manifest.json"
        if not manifest_path.is_file():
            raise SurrogateArtifactIntegrityError(
                f"Surrogate bundle manifest does not exist: {manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SurrogateArtifactIntegrityError(
                f"Cannot read surrogate bundle manifest: {exc}"
            ) from exc
        build_spec = manifest.get("build_spec")
        if not isinstance(build_spec, dict):
            raise SurrogateArtifactIntegrityError(
                "Artifact manifest does not contain a valid published build specification."
            )
        return cls.load(bundle_path, build_spec)

    @classmethod
    def publish(
        cls,
        bundle_path: str | Path,
        surrogate: TernaryMovingBoundaryThermodynamicsSurrogate,
        build_spec: Mapping[str, Any],
        members: Mapping[str, str | Path | Mapping[str, Any]] | None = None,
    ) -> "SurrogateArtifactBundle":
        """Atomically publish a model and diagnostic members to a bundle.

        Member values may be paths or dictionaries containing ``path`` and an
        optional ``schema_version``. Files are content-addressed and the
        manifest is replaced last, so a failed publication cannot advertise a
        partially written artifact.
        """

        bundle_path = Path(bundle_path)
        bundle_path.mkdir(parents=True, exist_ok=True)
        canonical_spec, fingerprint = surrogate_build_fingerprint(build_spec)
        surrogate.metadata["surrogate_artifact"] = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "build_fingerprint": fingerprint,
        }

        published: dict[str, dict[str, Any]] = {}
        temporary_paths: list[Path] = []
        try:
            handle, temporary_name = tempfile.mkstemp(prefix=".model-", suffix=".npz", dir=bundle_path)
            os.close(handle)
            temporary_model = Path(temporary_name)
            temporary_paths.append(temporary_model)
            surrogate.save(temporary_model)
            try:
                validation_model = TernaryMovingBoundaryThermodynamicsSurrogate.load(temporary_model)
            except Exception as exc:
                raise SurrogateArtifactIntegrityError(
                    f"Newly serialized surrogate could not be read back: {exc}"
                ) from exc
            if validation_model.metadata.get("surrogate_artifact", {}).get("build_fingerprint") != fingerprint:
                raise SurrogateArtifactIntegrityError(
                    "Newly serialized surrogate did not retain its build fingerprint."
                )
            model_digest = file_sha256(temporary_model)
            model_name = f"model-{model_digest}.npz"
            model_path = bundle_path / model_name
            if model_path.exists():
                if file_sha256(model_path) == model_digest:
                    temporary_model.unlink()
                else:
                    os.replace(temporary_model, model_path)
            else:
                os.replace(temporary_model, model_path)
            temporary_paths.remove(temporary_model)
            published["model"] = {
                "path": model_name,
                "size": model_path.stat().st_size,
                "sha256": model_digest,
                "schema_version": SURROGATE_SERIALIZATION_VERSION,
            }

            for role, value in (members or {}).items():
                descriptor = {"path": value} if isinstance(value, (str, Path)) else dict(value)
                source = Path(descriptor.pop("path"))
                descriptor.setdefault("schema_version", 1)
                if not source.is_file():
                    raise SurrogateArtifactIntegrityError(f"Artifact member {role!r} does not exist: {source}")
                expected_schema = descriptor.get("schema_version")
                if source.suffix.lower() == ".jsonl" and expected_schema is not None:
                    actual_schema = _jsonl_schema(source)
                    if actual_schema != expected_schema:
                        raise SurrogateArtifactIntegrityError(
                            f"Artifact member {role!r} has JSONL schema {actual_schema!r}; "
                            f"expected {expected_schema!r}."
                        )
                digest = file_sha256(source)
                safe_role = "".join(character if character.isalnum() or character in "-_" else "-" for character in str(role))
                name = f"{safe_role}-{digest}{source.suffix.lower()}"
                destination = bundle_path / name
                if not destination.exists() or file_sha256(destination) != digest:
                    handle, temporary_name = tempfile.mkstemp(prefix=f".{safe_role}-", suffix=source.suffix, dir=bundle_path)
                    os.close(handle)
                    temporary = Path(temporary_name)
                    temporary_paths.append(temporary)
                    shutil.copyfile(source, temporary)
                    if file_sha256(temporary) != digest:
                        raise SurrogateArtifactIntegrityError(f"Artifact member {role!r} changed while being copied.")
                    os.replace(temporary, destination)
                    temporary_paths.remove(temporary)
                published[str(role)] = {
                    "path": name,
                    "size": destination.stat().st_size,
                    "sha256": digest,
                    **descriptor,
                }

            manifest = {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": "ternary_moving_boundary_thermodynamics_surrogate",
                "surrogate_serialization_version": SURROGATE_SERIALIZATION_VERSION,
                "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "build_fingerprint": fingerprint,
                "build_spec": canonical_spec,
                "model_identity": {
                    "elements": list(surrogate.elements),
                    "phases": list(surrogate.phases),
                    "tieline_phases": list(surrogate.tieline_phases),
                    "temperature": float(surrogate.temperature),
                    "diffusivity_interpolation": surrogate.diffusivityInterpolation,
                },
                "members": published,
            }
            handle, temporary_name = tempfile.mkstemp(prefix=".manifest-", suffix=".json", dir=bundle_path)
            os.close(handle)
            temporary_manifest = Path(temporary_name)
            temporary_paths.append(temporary_manifest)
            temporary_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
            )
            os.replace(temporary_manifest, bundle_path / "manifest.json")
            temporary_paths.remove(temporary_manifest)
        finally:
            for temporary in temporary_paths:
                temporary.unlink(missing_ok=True)

        cls._remove_unreferenced_members(bundle_path, published)
        return cls(path=bundle_path.resolve(), surrogate=surrogate, manifest=manifest)

    @classmethod
    def load(
        cls,
        bundle_path: str | Path,
        expected_build_spec: Mapping[str, Any],
        required_members: Sequence[str] = (),
    ) -> "SurrogateArtifactBundle":
        """Load an intact artifact only when its build specification matches.

        This method is deliberately fail-closed. It never falls back to a raw
        NPZ archive and never invokes a builder when compatibility or integrity
        checks fail.
        """

        bundle_path = Path(bundle_path)
        manifest_path = bundle_path / "manifest.json"
        if not manifest_path.is_file():
            raise SurrogateArtifactIntegrityError(f"Surrogate bundle manifest does not exist: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SurrogateArtifactIntegrityError(f"Cannot read surrogate bundle manifest: {exc}") from exc
        if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise SurrogateArtifactIntegrityError(
                f"Unsupported artifact schema {manifest.get('artifact_schema_version')!r}; "
                f"expected {ARTIFACT_SCHEMA_VERSION}."
            )
        if manifest.get("artifact_type") != "ternary_moving_boundary_thermodynamics_surrogate":
            raise SurrogateArtifactIntegrityError(f"Unexpected artifact type {manifest.get('artifact_type')!r}.")
        if manifest.get("surrogate_serialization_version") != SURROGATE_SERIALIZATION_VERSION:
            raise SurrogateArtifactIntegrityError(
                f"Unsupported surrogate serialization version {manifest.get('surrogate_serialization_version')!r}."
            )

        canonical_spec, fingerprint = surrogate_build_fingerprint(expected_build_spec)
        saved_spec = manifest.get("build_spec")
        if manifest.get("build_fingerprint") != fingerprint or saved_spec != canonical_spec:
            details = _differences(canonical_spec, saved_spec)
            suffix = "" if not details else "\n  " + "\n  ".join(details)
            raise SurrogateArtifactCompatibilityError(
                "Saved surrogate is incompatible with the current build specification." + suffix
            )

        members = manifest.get("members")
        if not isinstance(members, dict) or "model" not in members:
            raise SurrogateArtifactIntegrityError("Artifact manifest does not define a model member.")
        missing_roles = sorted(set(str(role) for role in required_members) - set(members))
        if missing_roles:
            raise SurrogateArtifactIntegrityError(f"Artifact is missing required members: {missing_roles}.")
        for role, descriptor in members.items():
            if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
                raise SurrogateArtifactIntegrityError(f"Artifact member {role!r} has an invalid descriptor.")
            path = _safe_member_path(bundle_path, descriptor["path"])
            if not path.is_file():
                raise SurrogateArtifactIntegrityError(f"Artifact member {role!r} is missing: {path}")
            if path.stat().st_size != descriptor.get("size"):
                raise SurrogateArtifactIntegrityError(f"Artifact member {role!r} has an unexpected size.")
            if file_sha256(path) != descriptor.get("sha256"):
                raise SurrogateArtifactIntegrityError(f"Artifact member {role!r} failed its SHA-256 check.")
            expected_schema = descriptor.get("schema_version")
            if path.suffix.lower() == ".jsonl" and expected_schema is not None:
                actual_schema = _jsonl_schema(path)
                if actual_schema != expected_schema:
                    raise SurrogateArtifactIntegrityError(
                        f"Artifact member {role!r} has JSONL schema {actual_schema!r}; expected {expected_schema!r}."
                    )

        try:
            surrogate = TernaryMovingBoundaryThermodynamicsSurrogate.load(
                _safe_member_path(bundle_path, members["model"]["path"])
            )
        except Exception as exc:
            raise SurrogateArtifactIntegrityError(f"Cannot load the surrogate model member: {exc}") from exc
        identity = {
            "elements": list(surrogate.elements),
            "phases": list(surrogate.phases),
            "tieline_phases": list(surrogate.tieline_phases),
            "temperature": float(surrogate.temperature),
            "diffusivity_interpolation": surrogate.diffusivityInterpolation,
        }
        if identity != manifest.get("model_identity"):
            raise SurrogateArtifactIntegrityError("Loaded surrogate identity does not match the bundle manifest.")
        artifact_metadata = surrogate.metadata.get("surrogate_artifact", {})
        if artifact_metadata.get("build_fingerprint") != fingerprint:
            raise SurrogateArtifactIntegrityError("Loaded surrogate does not contain the manifest build fingerprint.")
        return cls(path=bundle_path.resolve(), surrogate=surrogate, manifest=manifest)

    @staticmethod
    def _remove_unreferenced_members(bundle_path: Path, members: Mapping[str, Mapping[str, Any]]):
        referenced = {descriptor["path"] for descriptor in members.values()} | {"manifest.json"}
        for path in bundle_path.iterdir():
            if path.is_file() and path.name not in referenced and not path.name.startswith("."):
                try:
                    path.unlink()
                except OSError:
                    # Cleanup is deliberately best-effort: the manifest has
                    # already been published and remains the source of truth.
                    pass


def prepare_surrogate_artifact(
    bundle_path: str | Path,
    build_spec: Mapping[str, Any],
    builder: Callable[[], tuple[TernaryMovingBoundaryThermodynamicsSurrogate, Mapping[str, Any]]],
    *,
    reload: bool = False,
    required_members: Sequence[str] = (),
) -> SurrogateArtifactBundle:
    """Reload a compatible bundle or build and publish a replacement.

    Rebuilding is the default. With ``reload=True`` the builder is never
    called: any missing, incompatible, or damaged artifact raises instead.
    """

    if reload:
        return SurrogateArtifactBundle.load(bundle_path, build_spec, required_members=required_members)
    surrogate, members = builder()
    return SurrogateArtifactBundle.publish(bundle_path, surrogate, build_spec, members)
