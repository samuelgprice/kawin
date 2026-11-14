# Package documentation conventions

These instructions supplement the repository-root instructions.

- Use repository-relative filesystem paths in documentation, examples of commands,
  and handoffs. State the starting directory explicitly: the extraction repository
  root or `packages/moving-boundary`. Keep Markdown links relative to their document.
- Name external sibling directories relative to that stated starting directory
  (for example, `../mb-public-050` from the extraction root). Resolve those paths
  before changing the working directory. Do not assume the same relative path works
  from both the repository root and the package directory.
- Keep user-specific absolute paths, usernames and workplace directory prefixes in
  ignored local configuration, such as `.verification/public-environment.json`,
  rather than committed documentation. Use a clearly labelled placeholder if an
  external location cannot be expressed relatively. Do not present `~` or a
  placeholder as a directly executable path without explaining how to resolve it.
- Established system/tool activation paths may remain absolute where required;
  preserve the repository's Conda activation rules. This convention does not require
  rewriting historical commits, generated provenance or scientific fixtures.
- Before finishing documentation changes, check the edited files for personal
  absolute paths and verify the stated working directory makes commands resolve.
