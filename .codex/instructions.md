# Project AI Instructions

## Git Commit Format

When an AI agent creates a git commit in this repository, the commit must use the
project format below.

Use the current repository git identity for AI-created commits:

```bash
git config user.name
git config user.email
```

Do not hard-code a committer or sign-off identity. If either value is missing,
ask the user to configure it before committing.

Commit message format:

```text
[Type]: Short summary

1.First concrete change.
2.Second concrete change.

Signed-off-by: <git config user.name> <git config user.email>
```

Allowed `Type` values:

```text
New
Fix
Update
Refactor
Docs
Test
Chore
```

Rules:

- Use the exact bracketed type prefix, for example `[Fix]: ...`.
- Keep the summary concise and specific.
- Use numbered body lines in the `1.Detail sentence.` style.
- Always include a `Signed-off-by` line generated from the active git config.
  Prefer `git commit -s` so git writes it automatically.
- If git identity is missing, do not invent or hard-code an identity. Ask the
  user to configure repository-local git identity, for example:

```bash
git config user.name 'Your Name'
git config user.email 'you@example.com'
```

- Stage and commit only files related to the current task.
- Do not include unrelated user changes in an AI-created commit.
