# qbench evidence report (SIMULATOR ONLY)

> **SIMULATOR ONLY.** Classical software running Qiskit Aer simulators with a synthetic noise model. No quantum hardware was used. Nothing here is a QEC, threshold, advantage or hardware claim.

- Plan: `qbench simulator demo: transpiler optimization level under a synthetic noise model` (schema `oes-resilience/qbench-plan/1`), SHA-256 `e0b5bffaa7c0e649ba7b7b08f4b800a94ec6134d0d78ab3c95b23315c2656c07`
- Lock: self-reported: plan SHA-256 recorded at 2026-10-08T01:44:52.636470+00:00 before the first run; plan file mtime 2026-10-08T01:44:44.872468+00:00; no external timestamp
- Seed 20261007, 2000 shots per run, families ghz, bv, qft, widths [3, 4, 5], 4 seed batches, 108 runs.

## Condition means

| condition | runs | hellinger_fidelity | total_variation_distance | success_probability |
|---|---|---|---|---|
| `ideal` | 36 | 1.0000 | 0.0022 | 1.0000 |
| `noisy_o1` | 36 | 0.8480 | 0.1520 | 0.8480 |
| `noisy_o3` | 36 | 0.8667 | 0.1333 | 0.8667 |

## Paired comparisons (block bootstrap, 95% percentile interval)

Differences are oriented so that a positive value favours condition b (for total variation distance, lower is better, so the sign is flipped).

| comparison | a | b | metric | blocks | mean diff | 95% CI | verdict |
|---|---|---|---|---|---|---|---|
| primary | `noisy_o1` | `noisy_o3` | hellinger_fidelity | 12 (family x batch) | +0.0187 | [+0.0053, +0.0324] | noisy_o3 better |
| primary | `noisy_o1` | `noisy_o3` | total_variation_distance | 12 (family x batch) | +0.0187 | [+0.0053, +0.0324] | noisy_o3 better |
| primary | `noisy_o1` | `noisy_o3` | success_probability | 12 (family x batch) | +0.0187 | [+0.0053, +0.0324] | noisy_o3 better |
| sanity | `noisy_o1` | `ideal` | hellinger_fidelity | 12 (family x batch) | +0.1520 | [+0.1126, +0.1918] | ideal better |
| sanity | `noisy_o1` | `ideal` | total_variation_distance | 12 (family x batch) | +0.1497 | [+0.1091, +0.1908] | ideal better |
| sanity | `noisy_o1` | `ideal` | success_probability | 12 (family x batch) | +0.1520 | [+0.1126, +0.1918] | ideal better |

## Metric-swap check

| comparison | primary verdict | flips across metrics? | verdicts |
|---|---|---|---|
| primary | noisy_o3 better | no | hellinger_fidelity: noisy_o3 better; total_variation_distance: noisy_o3 better; success_probability: noisy_o3 better |
| sanity | ideal better | no | hellinger_fidelity: ideal better; total_variation_distance: ideal better; success_probability: ideal better |

## Block-scheme sensitivity (primary metric)

| comparison | blocks | n blocks | mean diff | 95% CI | verdict | note |
|---|---|---|---|---|---|---|
| primary | family | 3 | +0.0187 | [+0.0000, +0.0538] | no significant difference | fewer than 5 blocks: interval unreliable |
| primary | batch | 4 | +0.0187 | [+0.0183, +0.0191] | noisy_o3 better | fewer than 5 blocks: interval unreliable |
| sanity | family | 3 | +0.1520 | [+0.0940, +0.2541] | ideal better | fewer than 5 blocks: interval unreliable |
| sanity | batch | 4 | +0.1520 | [+0.1502, +0.1538] | ideal better | fewer than 5 blocks: interval unreliable |

## What the hashes do and do not show

SHA-256 digests show only that a listed file is byte-for-byte identical to the recorded digest. They are not signatures and do not prove who produced a file, when it was produced, its provenance, or that a method or measurement is valid.

The plan hash is self-recorded before the first run; that is not an independent lock. Deposit the plan file with an external timestamp (for example Zenodo or OSF) before running for a credible lock.
