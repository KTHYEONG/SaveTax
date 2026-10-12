# Domain & System Core Directives

> **The primary directive is engineering reliable, reproducible systems that guarantee correctness, deterministic execution, and data integrity. Maintain strict boundaries and constitutional invariants across all application layers.**

## 1. System Ground Truth & Hard Boundaries
- **Causal Ordering & Consistency:** Data pipelines, event handlers, and core state transitions must strictly respect temporal causality and causal ordering.
- **Data Integrity & Conservation:** Accounting, balances, transactional states, and records must reconcile exactly without leakage or silent drop.
- **Zero Magic Numbers:** Never hardcode system thresholds, business limits, or domain parameters as raw literals in business logic. Parameters must be declared as typed, configurable contracts (`Spec`/`Config`).

## 2. Invariants & Deterministic Fail-Closed
- **Deterministic Fail-Closed:** On unrecoverable data anomalies, schema mismatches, or unexpected external failures, safely abort or reject state transitions rather than guessing or silently substituting default values.
- **Explicit Error Boundaries:** Define explicit error types and boundaries. Handle recoverable conditions gracefully and fail fast on invariant violations.

## 3. Modular Architecture & Evolution
- **Contract-First Public Boundaries:** Maintain clean separation between domain logic, data persistence, and external integration points.
- **Workspace Hygiene:** Keep exploratory experiments and temporary files strictly isolated under `scratch/`. Never commit raw cache or scratch files.
