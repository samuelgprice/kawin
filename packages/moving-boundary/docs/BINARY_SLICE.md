# Complete local solver and helper relocation

All three Illingworth solver classes and complete required local modules now live
in moving_boundary. They derive from public host DiffusionModel through bounded
package adapters, never from fork Illingworth classes. Shared private helpers live
in solvers/illingworth/shared.py. Host framework/engine ownership is described in
PUBLIC_KAWIN_MILESTONE.md.

Source authority for every row: samuelgprice/kawin
06dc7b4ab56f67226ceabfde41ebf2690c088841. Mechanical copies preserve numerical
AST, documentation, exception and unit conventions; import/module placement
changes and two whitespace-only binary edits are recorded. The older ae9fb29
fixture remains a separate oracle. No frozen source is developed or refreshed.

| Package path | Source path / extraction | Changes |
|---|---|---|
| solvers/illingworth/binary.py | kawin/diffusion/MovingBoundaryIllingworthFDM.py | module placement/imports only |
| solvers/illingworth/numerics/binary.py | kawin/diffusion/mesh/MovingBoundaryIllingworthFD1D.py | module placement/imports only |
| mesh/FDM1D.py | kawin/diffusion/mesh/FDM1D.py | module placement/imports only |
| mesh/TransformedGrids.py | kawin/diffusion/mesh/TransformedGrids.py | module placement/imports only |
| solvers/illingworth/_interstitials.py | kawin/diffusion/_illingworth_interstitials.py | module placement/imports only |
| solvers/illingworth/shared.py | kawin/diffusion/MovingBoundaryIllingworthTernaryFDM.py | complete pre-class helper section; imports only |
| solvers/illingworth/numerics/ternary.py | kawin/diffusion/mesh/MovingBoundaryIllingworthTernaryFD1D.py | module placement/imports only |
| equilibrium.py | kawin/diffusion/MovingBoundaryEquilibrium.py | module placement/imports only |
| _spectral_validation.py | kawin/diffusion/_spectral_validation.py | module placement/imports only |
| solvers/illingworth/ternary.py | kawin/diffusion/MovingBoundaryIllingworthTernaryFDM.py | class body unchanged; complete shared helper section moved to solvers/illingworth/shared.py |
| solvers/illingworth/three_phase.py | kawin/diffusion/MovingBoundaryIllingworthTernaryThreePhaseFDM.py | module placement/imports only |
| surrogates.py | kawin/diffusion/MovingBoundarySurrogates.py | Whole local provider module; imports and two inherited trailing-whitespace lines only; numerical AST identical |
| surrogate_artifacts.py | kawin/diffusion/SurrogateArtifacts.py | Whole local provider/helper module; imports only |
| _diffusivity_domain.py | kawin/diffusion/_diffusivity_domain.py | Whole local provider/helper module; imports only |

compatibility/ policy methods and their precise upstream-derived source locations
are mapped separately in PUBLIC_KAWIN_MILESTONE.md. No framework clone is hidden
in the helper copies. solvers/illingworth/numerics/tridiagonal.py re-exports the complete binary helper's kernel.

Binary 8233ad2, ternary e08eed0 and three-phase 693595d each passed 29 exact
installed-wheel run comparisons plus 22 tests before the dependency switch.
Quick: 29 run comparisons and 41 pytest tests passed in 435.73 s. Full: 38 run comparisons and 41 pytest tests passed in 495.45 s.
Independent exact/similarity errors accompany short/long drift checks. The suite
includes supported interstitial/three-phase volume combinations, irregular grid,
real pycalphad sampling, saved providers and configured terminal retry stopping.
Binary/two-phase unequal volumes and topology transitions are unsupported.

Upstream MIT text remains verbatim in NOTICE. The binary scheme identifies a port
of the authors' MAP C++ code; local/MAP rights remain unresolved. No C++ source,
PDFs, digitized data, databases or TC outputs enter the wheel. Numerical fixes
must have one authority and explicit source/port records. Package identity,
audience, publication, licensing outreach and prompt-default changes remain open.
