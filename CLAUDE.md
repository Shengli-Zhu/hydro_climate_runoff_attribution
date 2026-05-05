# Project rules for Claude Code

## Git / GitHub commit policy (MUST READ before every commit or push)

For this repository, **Claude must NEVER appear as an author, committer, or
co-author on any commit**. Specifically:

- **Never add a `Co-Authored-By: Claude ...` trailer** to any commit message.
- **Never include any `Co-Authored-By: ...@anthropic.com`** trailer.
- **Never include "🤖 Generated with [Claude Code]"** or any equivalent
  attribution line in commit messages.
- **Never run `git commit` with `--author` set to anything other than
  the user's own identity** (`Shengli-Zhu <shengli.zhu@kaust.edu.sa>`).
- **Never modify** `git config user.name` or `git config user.email`.

Before running `git commit`, verify:

1. The HEREDOC body of the commit message contains no "Claude", "Anthropic",
   or "Co-Authored-By" lines.
2. `git config user.name` is `Shengli-Zhu` and `git config user.email` is
   `shengli.zhu@kaust.edu.sa` (do not change them).

Before running `git push`, verify with:

```bash
git log <new-commits> --pretty=format:'%an <%ae> | %B' | grep -iE "claude|anthropic|co-author"
```

If the grep returns any match, **do not push**. Re-create the commit with
a clean message instead.

## Why

The user has had to delete and rebuild this repository once because a
"claude Claude" entry appeared in the GitHub Contributors widget. This must
not happen again.
