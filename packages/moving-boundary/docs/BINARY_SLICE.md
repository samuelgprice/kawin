# Complete Illingworth module relocation

The binary solver is package-owned and directly derives from the host
DiffusionModel. It no longer inherits the fork's Illingworth class. Complete local
modules replace the earlier two-method subclass checkpoint. The host model/solver
loop, mesh bases, MeshData and thermodynamic engine remain dependency-owned.

Every entry below comes from samuelgprice/kawin at
06dc7b4ab56f67226ceabfde41ebf2690c088841. The older numerical fixture remains
ae9fb29, and the temporary installed host pin is the test-only cleanup 5f4ad57.
The detached 06dc7b4 source stays clean. Source files are copied mechanically;
imports/module placement change, preserving arithmetic, docstrings, exception and
boundary conventions, temperature-array hours and supported composition bases.

| Package path | Source path / extraction | Changes |
|---|---|---|
| solvers/binary.py | kawin/diffusion/MovingBoundaryIllingworthFDM.py | module placement/imports only |
| numerics/binary.py | kawin/diffusion/mesh/MovingBoundaryIllingworthFD1D.py | module placement/imports only |
| mesh/FDM1D.py | kawin/diffusion/mesh/FDM1D.py | module placement/imports only |
| mesh/TransformedGrids.py | kawin/diffusion/mesh/TransformedGrids.py | module placement/imports only |
| _illingworth_interstitials.py | kawin/diffusion/_illingworth_interstitials.py | module placement/imports only |
| shared.py | kawin/diffusion/MovingBoundaryIllingworthTernaryFDM.py | complete pre-class helper section; imports only |
| numerics/ternary.py | kawin/diffusion/mesh/MovingBoundaryIllingworthTernaryFD1D.py | module placement/imports only |
| equilibrium.py | kawin/diffusion/MovingBoundaryEquilibrium.py | module placement/imports only |
| _spectral_validation.py | kawin/diffusion/_spectral_validation.py | module placement/imports only |
| solvers/ternary.py | kawin/diffusion/MovingBoundaryIllingworthTernaryFDM.py | class body unchanged; complete shared helper section moved to shared.py |
| solvers/three_phase.py | kawin/diffusion/MovingBoundaryIllingworthTernaryThreePhaseFDM.py | module placement/imports only |

The scalar kernel's old numerics/tridiagonal.py path now re-exports from the whole
binary helper module, avoiding two independent copies. Shared private helpers
belong in shared.py instead of being imported from a ternary solver. See
relocation-map.json for machine-readable source mapping.

The mandatory quick gate runs every approved short case, independent analytic
errors, real pycalphad sampling, old binary fixtures/bulk probes and package
pytest contracts against privately installed wheels in separate processes.
Each family under relocation must originate in the installed package. Families
not yet moved are explicitly reported as host-backed intermediates. Per-stage
exact parity and counts are in acceptance.json; full is rerun at the public gate.

Binary/two-phase unequal molar volumes remain unsupported; three-phase explicit
volume semantics are retained. C32 tests configured terminal-retry stop under
continue without a fake solve callback. No topology-transition support is added.

The fork identifies the binary scheme as a port of the authors' MAP C++ code.
Upstream MIT text remains verbatim in NOTICE; local/MAP rights remain unresolved.
No C++ reference source, PDFs, digitized data or TC datasets enter this wheel.
Scientific fixes must have one authoritative source and an explicitly recorded
port; the frozen reference is not developed independently to hide discrepancies.
Package identity/audience, publication and licensing outreach remain user decisions.
