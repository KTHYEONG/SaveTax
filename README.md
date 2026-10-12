# SaveTax

SaveTax core system foundation.

## Development Setup

```bash
# Sync virtual environment & install dependencies
uv sync

# Run verification gate (lint, mypy, pytest, diff-coverage)
uv run python tools/verify.py

# Run pytest directly
uv run pytest
```

## Candidate Screening

    uv run python -m src.savetax.cli screen --batch 0 --handle <public-handle>
    uv run python -m src.savetax.cli screen --batch 0 --batches 3 --online

## Evidence Collection

    uv run python -m src.savetax.cli evidence --lane cross --limit 50
    uv run python -m src.savetax.cli evidence --lane cross --items 25051267
