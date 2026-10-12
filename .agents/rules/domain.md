# Domain & System Core Directives: SaveTax Hackathon

> **The primary directive is engineering an objective, deterministic, and evidence-backed public procurement audit engine to detect inflated government spending (세금도둑 적발). Maintain strict mathematical invariants on price delta computations, absolute protection of participant credentials, and rigorous truth preservation.**

## 1. Domain Ground Truth & Calculation Invariants
- **Procurement vs. Market Price Invariants:**
  - `procurement_unit_price > 0`, `market_reference_price > 0`, `contract_quantity > 0`.
  - `price_delta = procurement_unit_price - market_reference_price`.
  - `tax_waste_amount = price_delta * contract_quantity`.
  - A candidate is an audit anomaly only when `procurement_unit_price > market_reference_price * markup_threshold` with identical or functionally equivalent specifications.
- **Zero Magic Numbers:** Configurable parameters (markup threshold multiplier, minimum total contract expenditure, recency filters) must be declared in strongly-typed configuration contracts (`AuditConfig`/`Spec`), never hardcoded as raw magic literals.
- **Absolute Truth Preservation (No Hallucination):** Never fabricate product IDs, delivery histories, market prices, or markup justifications. If datasets, APIs, or local caches are unreachable, halt candidate exploration with explicit status `자료 접근 대기` rather than fabricating evidence or claiming completion.
- **Causal Data Ordering:** Candidate evaluation must trace historical procurement records chronologically against contemporaneous market prices at delivery time.

## 2. Participant Credential & Security Invariants
- **PAT Isolation Boundary:** Participant Personal Access Tokens (PAT) are strictly confined to:
  1. Protected SaveTax API (`api.savetax.now`) via `Authorization: Bearer <PAT>` header.
  2. The single official product ZIP URL (`https://savetax-dataset.reform-dev-infra.workers.dev/latest/savetax-products.zip`) via `Authorization: Bearer <PAT>` header.
  3. The official SaveTax MCP (`https://mcp.savetax.now/mcp`) via `Authorization: Bearer <PAT>` header, registered with `claude mcp add --scope local` so the token never lands in repository files (`.mcp.json` must not carry it).
- **Prohibited Leakage Vectors:** NEVER pass PAT to:
  - Public manifest or documentation URLs (`hackathon.savetax.now`).
  - Arbitrary `workers.dev` endpoints outside the single official ZIP URL.
  - URL query strings, command-line arguments, shell histories, logs, console output, code, commit messages, or persistent repository files.
  - DuckDB query strings, random seeds, or partition identifiers.

## 3. Network & Environmental Diagnostic Rigor
- **Multi-Stage Connection Diagnostics:** When probing APIs or dataset endpoints, independently verify and log:
  1. DNS Resolution (Host IP lookup).
  2. TCP Handshake & TLS Negotiation.
  3. HTTP Response Status.
- **Sandbox vs. Infrastructure Discrimination:** Distinguish local OS/container sandbox restrictions (`Operation not permitted`, socket creation denied) from remote server outages. Never attribute sandbox blocks or pre-HTTP failures to host DNS/server crashes.
- **Transient Failure Handling:** On transient connection errors, execute exactly one retry after a 15-second cooldown. If blocked by network permissions, request official platform approval with the specific host required.

## 4. Submission & Reporting Boundaries
- **Autonomous General Reports (일반 제보):** Candidates with robust, verified market price evidence (Danawa MCP / verified catalogs) and exact product spec matching may be reported autonomously via the `reports` API without blocking for human confirmation.
- **Reviewer Feedback Feedback-Loop:** Continually ingest review outcomes from the `reports` API. Parse `review_code`, `reviewer_note`, and evaluation history from rejected reports to refine evidence criteria for future candidates.
- **Strict User Approval Gate (최종 점수 제출):**
  - Final score submission (`score submit`), modifications, or withdrawals REQUIRE explicit user confirmation.
  - Modifying or re-submitting previously rejected reports requires participant confirmation.
