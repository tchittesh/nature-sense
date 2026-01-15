# Project Instructions for Claude

## Git Workflow

**NEVER create git commits without explicit user permission.**

Only commit when the user specifically asks you to commit changes.

## Python Environment

This project uses a conda environment. Always activate it before running Python commands:

```bash
conda activate nature_observer
```

When running Python scripts or commands, prefix with `conda run -n nature_observer` or activate the environment first.

## Code Style


### Python

1. Prefer `from ... import ...` style except for clear conventions like `import numpy as np`.
2. Use type hints!
