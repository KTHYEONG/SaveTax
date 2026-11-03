---
paths:
  - "src/**"
---
# Performance & Optimization Directives: SaveTax Hackathon

> **Performance is an architectural invariant, not an afterthought. Never fake speed by silently omitting candidate batches, truncating catalog searches, or skipping evidence validation. Correctness strictly precedes speed. Maximize throughput across dataset exploration and candidate auditing within physical boundaries.**

## 1. Bounded Resource Scaling & Memory Discipline
- **Bounded Resource Scaling:** Memory consumption must remain safely bounded within available host RAM. When processing the large official `savetax-products.zip` or full Parquet tables, use streaming, chunked reading, or DuckDB disk-backed/view-based processing rather than loading entire multi-gigabyte tables into uncompressed Python memory.
- **Eliminate Unnecessary Duplication:** Avoid redundant in-memory DataFrame copies, intermediate JSON serializations, and repeated regex compilations.
- **Deterministic Resource Reclamation:** Explicitly close DuckDB connections, zip file streams, network response iterators, and temporary file handles upon exiting operational scopes.

## 2. Algorithmic Locality & Hot Path Efficiency
- **Hot Path Algorithmic Efficiency:** When comparing thousands of procurement items against market price records, eliminate nested linear scans ($O(N \times M)$ traps). Pre-index catalogs using hash maps, standardized model codes, or DuckDB relational joins.
- **I/O Locality & Column Pruning:** Never read unused metadata columns during full-catalog scans. Leverage DuckDB columnar projection pushdown (`SELECT id, name, price, qty...`) and predicate filtering (`WHERE price > min_threshold`) to minimize disk and bus traffic.

## 3. Candidate Exploration & Seeded Shuffling
- **Participant-Seeded Batching:** In DuckDB full catalog scans, generate deterministic pseudorandom batches using the competition API's public handle and dataset version as seeds. Ensure reproducible, non-overlapping batch traversals.
- **Early Rejection Heuristics:** Filter out obvious non-anomalies early in the pipeline (e.g., procurement prices below market baseline, tiny contract quantities with negligible waste) before triggering expensive external Danawa MCP searches or full specification parsing.

## 4. Hardware Saturation & Observable Throughput
- **Balanced Parallelism:** Saturate available CPU cores for compute-heavy specification parsing or batch anomaly scoring. Ensure batch sizes are large enough to amortize worker communication costs.
- **Continuous Progress Telemetry:** Long-running dataset ingestion or batch audit scans must emit periodic progress heartbeats (e.g., every 5,000 items or 10 seconds). A silent audit pipeline is indistinguishable from a deadlocked process.