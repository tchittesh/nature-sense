# Project Instructions for Claude

## Git Workflow

**NEVER create git commits without explicit user permission.**

Only commit when the user specifically asks you to commit changes.

Before creating any commit, run `conda run -n nature_sense prek run --all-files` and fix anything it reports. Do not commit until all hooks pass.

## Python Environment

This project uses a conda environment. Always activate it before running Python commands:

```bash
conda activate nature_sense
```

When running Python scripts or commands, prefix with `conda run -n nature_sense` or activate the environment first.

## Scripts

Keep the Scripts table in `README.md` in sync whenever you add, remove, or significantly change a script — each entry should be a 1-2 line summary of what the script does.

## Dependencies

**Always add new dependencies to `requirements.txt`** when introducing them in code. Do not install packages without recording them there. Pin dependencies to the exact installed version (e.g. `modal==1.4.1`, not `modal`).

## Code Style

- Prefer verbose but descriptive variable names e.g. `object_points` instead of `obj_pts`.

### Python

- Include `from __future__ import annotations` to deal with self-referential type hints.
- Prefer `from ... import ...` style except for clear conventions like `import numpy as np`.
- Use type hints!
- Prefer `if X is not None` over `if X` to prevent logic errors on falsey values.
