"""Binary Illingworth facade requiring the documented research fork.

The exported class is the original class, with unchanged signatures, docstrings,
timestep policy, physical conventions and result types. Mesh and material inputs
still come from that fork. Ternary facades are deferred.
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
        from kawin.diffusion.MovingBoundaryIllingworthFDM import (
            MovingBoundaryIllingworthFD1DModel,
        )
    except ModuleNotFoundError as exc:
        if exc.name in {"kawin", "kawin.diffusion",
                        "kawin.diffusion.MovingBoundaryIllingworthFDM"}:
            raise ImportError(
                "This development bridge requires samuelgprice/kawin at "
                "ae9fb2971a9fd49132f71ddc883a5bd5406644b6; public kawin "
                "does not provide this solver. See the package README."
            ) from exc
        raise
    return MovingBoundaryIllingworthFD1DModel
