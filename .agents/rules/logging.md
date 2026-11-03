---
paths:
  - "src/**"
---
# Unified Logging & Diagnostic Directives: SaveTax Hackathon

> **Logs must be concise, structured, and machine-parsable without sacrificing diagnostic correctness. Never log secret credentials (PAT), and strictly isolate network diagnostic stages to prevent false infrastructure alarms.**

## 1. Information Integrity & Credential Security
- **Strict Credential Redaction:** NEVER log participant Personal Access Tokens (PAT), Authorization headers, session cookies, or private credentials in plaintext or truncated forms.
- **Precision Preservation:** Retain exact numerical values for currency, unit prices, markup multipliers, and total tax waste amounts. Never round significant price differences to coarse approximations.
- **Stable Correlation Identifiers:** Include stable candidate identifiers (`product_id`, `contract_num`, `audit_batch_id`, `report_id`) across multi-stage exploration pipelines.

## 2. Standard Category Taxonomy & Structured Output
- **Fixed Domain Category Taxonomy:** Classify all operational log events using one of six fixed category tags:
  - `[SYS]`: Host runtime, network diagnostics (DNS, TCP/TLS, HTTP stages), sandbox permission states, and periodic guidance polling (3-hour interval).
  - `[DATA]`: Dataset acquisition (official ZIP streaming), DuckDB Parquet batch scanning, table schemas, and procurement catalog ingestion.
  - `[AUDIT]`: Price anomaly detection, markup ratio evaluation, unit price delta calculations, and candidate filtering heuristics.
  - `[EVIDENCE]`: Market price cross-verification (Danawa MCP tool queries), product specification matching, and evidence bundle generation.
  - `[REPORT]`: Hackathon reports API submissions, candidate status transitions, and rejected review analysis (`review_code`, `reviewer_note`, history).
  - `[SUBMIT]`: Final competition score submissions, modifications, and user-approval gate confirmations.
- **Consistent Log Format:** Operational events must use the category prefix:
  `logger.info("[<CATEGORY>] <event description>")`
- **Structured Fields:** Prefer structured key-value pairs or JSONL records over conversational text:
  `logger.info("[AUDIT] Candidate flagged product_id=%s procurement_price=%d market_price=%d markup=%.2fx waste=%d", ...)`
- **Large Dataset Summaries:** Never dump entire raw Parquet tables or DuckDB result sets into logs. Summarize queries by row count, column schema, price delta quantiles, or top-N anomaly candidate summaries.

## 3. Network Diagnostics & Failure Logging
- **Granular Multi-Stage Connection Logging:** When logging connection failures to `api.savetax.now`, `hackathon.savetax.now`, or dataset workers:
  - Log distinct stages: `DNS_RESOLUTION`, `TCP_TLS_CONNECT`, `HTTP_STATUS`.
  - Distinguish sandbox errors (e.g. `[SYS] Network blocked: Operation not permitted (local sandbox policy)`) from remote host unreachability.
- **Data Unavailability Protocol:** When data sources are completely unreachable after retries, log with explicit status `[SYS] Candidate search paused: 자료 접근 대기 (Reason: <stage_failure>, Resume condition: <condition>)`.

## 4. Lifecycle & Hygiene
- **Configured Storage Paths:** Operational logs must reside in designated project log paths. Ephemeral exploration traces belong under `scratch/`.
- **Non-Destructive Cleanup:** Never unilaterally delete audit logs or historical evidence needed to verify submitted reports.
