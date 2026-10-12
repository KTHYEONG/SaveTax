# Code Style & Engineering Standards: SaveTax Hackathon

> **Respect existing architecture and conventions. Maintain strong typed contracts at public boundaries without breaking scope. Reconcile code with specifications objectively, and apply equivalent engineering rigor across all modules.**

## 1. Architectural Integrity & Modularity
- **Layout & Scope Hygiene:** Respect the repository's source/test layout (`src/savetax/`, `tests/unit/`). Maintain high cohesion between data ingestion, price auditing, market verification, and API reporting layers.
- **Contract & Spec Reconciliation:** Treat validated executable code, Pydantic schemas, and API contracts as authoritative for implementation. Actively investigate discrepancies with specifications rather than assuming either side is automatically correct.

## 2. Typing, Contracts & Configuration
- **Strong Typing at Boundaries:** Enforce strict static typing (`mypy --strict`) for all public interfaces, data models, and audit functions. Never use unconstrained `Any` when a typed Model, TypedDict, or Generic can be expressed.
- **Explicit Configuration:** Externalize all endpoint URLs, timeout thresholds, markup multiplier limits, and participant handles in strongly-typed configuration contracts (`SaveTaxConfig`).
- **Toolchain Alignment:** Use `uv`, `ruff`, `mypy`, and `pytest` via `tools/verify.py`. Do not introduce competing tools.

## 3. Resilience & Defensive Boundaries
- **Network Client Robustness:** HTTP clients communicating with `api.savetax.now` or dataset workers must implement explicit timeouts, separate DNS/connection error tracking, User-Agent headers (`Mozilla/5.0`), and single 15-second retry backoffs for transient errors.
- **Fail-Closed on Corrupted Data:** When encountering corrupt Parquet chunks, invalid JSON schemas, or malformed procurement records, fail closed or route to quarantine logging rather than silently injecting dummy values.
