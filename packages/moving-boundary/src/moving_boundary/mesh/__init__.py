"""Local finite-difference additions with host mesh bases/boundaries retained."""
from .FDM1D import CartesianFD1D, FiniteDifference1D
from kawin.diffusion.mesh import MixedBoundary1D, PeriodicBoundary1D, ProfileBuilder, StepProfile1D
