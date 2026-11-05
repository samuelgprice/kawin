# Next three sessions

Updated November 5, 2026. This execution list governs the next steps; the long
EXTRACTION_PLAN.md remains background. No driver rewrite is authorized by a
packaging failure alone.

1. **Verification and bounded host assessment completed.**
   The summary exposes the installed-fork gate; strict mode exits nonzero.
   Eleven package/verification tests and all nine substitutional cases pass.
   The public 0.5.0 wheel was inspected, not installed. See LEDGERS.md for the
   comparison and source hash. Localized host differences favor retaining the
   host rather than starting a complete driver rewrite.

2. **Remove production debugger dependencies in a narrow, separate fork change.**
   Keep this reference frozen. Cover all import/reload paths, including surrogates
   and Olaye, and ternary/three-phase retry exhaustion. Preserve exceptions,
   exception chaining and accepted state. Verify installed imports with no source
   tree plus failing-step tests without debugger stubs or notebook waits.
   Selected interstitial/retry/rollback prerequisites now pass (39 tests);
   complete public failure-path coverage remains required for this change.
   Give the maintained fork change a distinct source reference before updating
   package dependency metadata. Do not replace the driver/results for this fix.
   Final package identity, audience and any upstream/licensing outreach remain
   user decisions; no messages have been sent.

3. **Implement one chosen binary slice.**
   Preserve source mapping, attribution, operation order, constants and units.
   Run isolated reference/extracted processes against frozen fixtures. Require
   installed-package imports without source checkout paths for any installation
   claim. Keep host driver/results unless the preceding assessment justifies
   replacing a bounded part. Do not extract all three solvers in one step.

## Baseline and concurrent research policy

- The reference is commit ae9fb2971a9fd49132f71ddc883a5bd5406644b6.
  Keep kawin/, its required example imports, fixtures and packaging inputs frozen
  in this extraction worktree. New verification and API work lives in packages/.
- This is not a universal requirement to copy every dependency. If a maintained
  fork fix is selected, implement it separately with its own SHA and evidence;
  retain the original reference for comparison and update dependency metadata
  explicitly. Do not silently patch the reference or use import monkeypatches.
- During extraction, scientific fixes are authored once on the active research
  branch (currently feature/DICTRA-comparison, or its designated successor).
  Port required fixes deliberately to extracted code. Record source commit,
  affected functions, reason, tests and expected numerical changes in a fix
  ledger. Packaging-only fixes belong to extraction. Never automatically merge
  research HEAD or regenerate fixtures to hide a numerical change.
- A baseline refresh is a separate reviewed scientific change. After release,
  explicitly transfer numerical authority to the new package and adapt research
  consumers; avoid indefinite independent implementations.
- Preserve floating-point update order (including derivative/step round trips),
  gas constant 8.3145 where used, and composition/flux conventions. Temperature
  arrays use hours; do not generalize that to every callback's time argument.
- Investigate MAP-port provenance before publication. No permissions, legal
  clearance, package-name availability or upstream acceptance is assumed.