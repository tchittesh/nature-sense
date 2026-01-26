# Project Instructions for Claude

## Git Workflow

**NEVER create git commits without explicit user permission.**

Only commit when the user specifically asks you to commit changes.

## Python Environment

This project uses a conda environment. Always activate it before running Python commands:

```bash
conda activate nature_sense
```

When running Python scripts or commands, prefix with `conda run -n nature_sense` or activate the environment first.

## Code Style

- Prefer verbose but descriptive variable names e.g. `object_points` instead of `obj_pts`.

### Python

- Include `from __future__ import annotations` to deal with self-referential type hints.
- Prefer `from ... import ...` style except for clear conventions like `import numpy as np`.
- Use type hints!
- Prefer `if X is not None` over `if X` to prevent logic errors on falsey values.
