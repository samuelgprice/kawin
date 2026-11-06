"""Binary Illingworth facade requiring the documented research fork.

The exported class inherits the host API and overrides only the two planar bulk
concentration solves, using an extracted tridiagonal kernel. Mesh, material,
timestep and result services still come from the fork. Ternary APIs are deferred.
"""

__all__ = ["MovingBoundaryIllingworthFD1DModel"]


def __getattr__(name):
    """Resolve the binary class lazily, with a clear missing-fork error.

    Importing the legacy submodule also executes kawin.diffusion initialization;
    this bridge does not yet isolate its transitive backend dependencies.
    """
    if name != "MovingBoundaryIllingworthFD1DModel":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from .binary import (
            MovingBoundaryIllingworthFD1DModel,
        )
    except ModuleNotFoundError as exc:
        if exc.name in {"kawin", "kawin.diffusion",
                        "kawin.diffusion.MovingBoundaryIllingworthFDM"}:
            raise ImportError(
                "This development bridge requires samuelgprice/kawin at "
                "06dc7b4ab56f67226ceabfde41ebf2690c088841; public kawin "
                "does not provide this solver. See the package README."
            ) from exc
        raise
    return MovingBoundaryIllingworthFD1DModel
