# Documentation & Code Commenting Directives: SaveTax Hackathon

> **Explain the non-obvious "Why" behind audit heuristics and system constraints, never the mechanical "What" visible in code. Keep code self-documenting, comments minimal, and eliminate narrative step-by-step bloat.**

## 1. Commenting Principles: The "Why" Standard
- **Self-Documenting Code First:** Code structure, expressive identifier names, and strong types must communicate what the code does. The default expectation is near-zero inline comments.
- **Rationale-Only Standard:** Write comments exclusively for non-obvious design trade-offs, regulatory/competition rules (e.g., Cloudflare 1010 User-Agent workarounds, HEAD vs GET nuances, sandbox error discrimination), and mathematical edge cases.
- **No Procedural Narration:** Omit mechanical paraphrasing, procedural steps, and session metadata.

## 2. Interface Documentation Standards
- **Compact Caller Contracts:** Docstrings are compact (1–2 lines) caller contracts describing inputs, return guarantees, and raised errors.
- **Contract-Driven Documentation:** Document public API clients, DuckDB query interfaces, and anomaly detection functions where units (e.g., KRW, percentage, multiplier) or edge-case handling are not self-evident from type signatures.

## 3. Architecture & Domain Documentation (`docs/`)
- **Stable Boundaries & Semantics:** Keep architecture and guide documents focused on stable system boundaries, data ingestion flows, candidate evaluation pipelines, and submission contracts.
- **Coherent In-Place Updates:** Update existing documentation coherently rather than accumulating disconnected notes.

## 4. Repository Language Conventions
- **All Documentation & In-line Comments:** Written in concise Technical English for toolchain compatibility, token efficiency, and deterministic agent reasoning.
