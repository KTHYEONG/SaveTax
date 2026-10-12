# Operating Manual: SaveTax

> **Domain Identity:** 2026 Reform Party Tax Thief Detection Hackathon (개혁신당 세금도둑 찾는 해커톤) - Automated Public Procurement Audit & Whistleblowing Engine.

## 1. Domain Ground Truth & Hard Boundaries
- **Objective & Integrity:** Detect and report excessively inflated public procurement goods (세금 낭비/과다 집행 품목) with verified empirical market price evidence. Never fabricate, extrapolate, or cosmetically embellish candidates. If data is inaccessible, mark as `자료 접근 대기`.
- **Absolute Credential Protection (PAT Invariant):**
  - The participant PAT must ONLY be transmitted via `Authorization: Bearer <PAT>` to protected SaveTax APIs (`api.savetax.now`), the official SaveTax MCP (`https://mcp.savetax.now/mcp`, registered via `claude mcp add --scope local --header`, never in repo files), and the official ZIP URL (`https://savetax-dataset.reform-dev-infra.workers.dev/latest/savetax-products.zip`).
  - NEVER output, log, hardcode, commit, or pass PAT via URL query parameters, CLI logs, seeds, queries, or public manifest requests.
- **REST & Dataset Driven Discovery:** Explore procurement data exclusively via REST APIs and official datasets (DuckDB/Parquet/ZIP). Never use browser automation, Chrome, or generic web searches for catalog discovery.
- **Network Diagnostic Rigor:** Separate connection failures across DNS, TCP/TLS, and HTTP stages. Distinguish local sandbox restrictions (`Operation not permitted`, socket rejections) from server outages. Never assume a host is down before reaching the HTTP layer.
- **Submission Autonomy Contract:**
  - **Regular Reports (일반 제보):** Autonomously submit well-substantiated candidate reports without per-candidate user confirmation. Periodically monitor reviewer feedback (`reports` API) to refine evidence quality.
  - **Score Submission & Modification (점수 제출/수정/철회):** Strictly gated by explicit user approval before execution. Modifying or re-submitting rejected reports also requires user confirmation.
- **Session-Based Guidance Polling:** Poll `https://hackathon.savetax.now/docs/agent-guidance-version.json` (unauthenticated) upon session start and every 3 hours during active work. No standalone daemon/cron scripts.

## 2. Autonomy & Diagnostic Contract
- **Bias Toward Action:** Autonomously run exploratory queries, DuckDB data batch probes, and candidate validations under `scratch/`.
- **Skills as Modular Utilities:** Skills (`probe`, `spec`, `implement`, `check`, `refactor`, `commit`) are independent utilities. Specs reside under `docs/specs/` (gitignored).

## 3. Project Toolchain & Verification
- **Quality Gate:** `uv run python tools/verify.py`
- **Test Runner:** `uv run pytest`
- **Git Commits:** Run the project's `commit` skill with Korean subject and 1-line rationale.

## 4. Communication & Language
- **Natural Korean:** Converse, explain findings, candidate diagnoses, and reports in Korean (한국어). Inside structured output cards, retain English keys/badges while writing descriptions in Korean.
- **Technical English:** System instructions, rules, specs, code, and docstrings are written in Technical English.

## 5. Domain Rule Routing
- **Domain Invariants:** [domain.md](.agents/rules/domain.md) — *Procurement tax audit integrity, market price delta invariants, and PAT protection.*
- **Testing & Quality:** [testing.md](.agents/rules/testing.md) — *Evidence verification, boundary tests, synthetic datasets, and diff-coverage.*
- **Architecture & Standards:** [code-style.md](.agents/rules/code-style.md) — *Data pipelines, typing contracts, and toolchain alignment.*
- **Documentation & Comments:** [documentation.md](.agents/rules/documentation.md) — *Contract docstrings, evidence rationale, and Technical English.*
- **Performance & Optimization:** [performance.md](.agents/rules/performance.md) — *DuckDB full-dataset scanning, memory discipline, and ZIP batch processing.*
- **Logging & Diagnostics:** [logging.md](.agents/rules/logging.md) — *Audit taxonomy, network diagnostic stages, and credential redaction.*
