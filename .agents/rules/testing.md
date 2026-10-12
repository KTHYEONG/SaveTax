# Testing Directives & Quality Standards: SaveTax Hackathon

> **Verify observable audit behaviors and data contracts, not internal implementation details. Maintain fast feedback loops with small deterministic synthetic procurement catalogs. Enforce invariant verification on price anomaly detection, isolate failures with pinpoint precision, and guarantee 100% diff-coverage.**

## 1. Development Paradigm: Invariant-Driven Development (IDD)
- **Contracts Over Premature Tests:** Prioritize strict static typing, Pydantic/dataclass schema contracts, and domain invariants over dogmatic test-first rituals.
- **Pragmatic Invariant Verification (Zero Ceremony):**
  - *Direct Execution (Bug fixes, anomaly heuristics, probe scripts):* Implement directly with targeted invariant guard tests and verify via `tools/verify.py`.
  - *Contract Design (Complex data pipelines, API clients, DuckDB parsers):* Formulate explicit interface contracts and invariant scenarios before implementation.
  - *On-Demand Utilities:* Skills (`probe`, `spec`, `implement`, `check`) are independent tools, not a mandatory sequential conveyor belt.
- **Observable Behavior & Invariants:** Verify return contracts, price delta calculations, data deduplication, and error conditions rather than private implementation details.
- **In-Memory & Synthetic Data:** Use minimal deterministic synthetic procurement and market records (small JSON/Parquet fixtures) in unit tests; never download the real full ZIP dataset in unit test runs.

## 2. Domain Invariants & Boundary Testing
- **Tax Waste & Price Delta Invariants:**
  - Verify that negative prices, zero quantities, or invalid currency types raise explicit validation errors (`ValidationError` or custom domain exceptions).
  - Verify exact integer or decimal parity on `tax_waste_amount = (procurement_price - market_price) * quantity`.
  - Verify markup ratio thresholds properly flag true anomalies and suppress noise.
- **Evidence & Specification Matching Boundaries:**
  - Test edge cases in product spec matching: model number variations, unit discrepancies (e.g., box vs. each), missing attributes, and ambiguous catalog descriptions.
- **Network Diagnostic & Resilience Boundaries:**
  - Unit-test error classification for DNS failures, TLS negotiation timeouts, HTTP 401/403/404/500 responses, and sandbox socket restrictions (`EPERM`/`Operation not permitted`).
  - Verify single 15-second retry logic for transient errors.

## 3. Execution, Isolation & Hermetic Environment
- **Hermetic Environment Invariant:** Unit tests must be 100% hermetic and isolated from live shell credentials (`SAVETAX_TOKEN`, participant PAT) and real external endpoints.
- **Pragmatic Fixtures & Mocking:**
  - Mock external endpoints (`api.savetax.now`, `hackathon.savetax.now`, Danawa MCP, Cloudflare Workers).
  - Never mock internal domain audit heuristics, mathematical calculations, or DuckDB queries when SQLite/in-memory DuckDB can run deterministically in-memory.
- **Process-Isolated Temp Workspaces:** Each test runner invocation operates in its own partitioned, process-isolated temporary workspace directory (`tmp/verify/...`) to prevent race conditions.

## 4. Diff-Coverage & Quality Philosophy
- **Diff-Coverage & Domain Focus:** Focus test coverage on newly-added audit algorithms, parser logic, and boundary conditions. All added lines must meet 100% diff-coverage via `tools/verify.py`.
- **Test Integrity:** Never weaken assertions, delete valid test cases, or bypass verification gates. Diagnose and remediate root causes directly.