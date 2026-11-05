# SMAP/MSL result verification record

## Status

- **Checksum manifest:** present at `reports/smap_msl_results/SHA256SUMS`.
- **Covered artifacts:** `smap_msl_comparisons.csv`, `smap_msl_event_f1.svg`, `smap_msl_per_channel.csv`, `smap_msl_results.json`, and `smap_msl_summary.csv`.
- **Local SHA-256 recomputation:** **PENDING**. Run `scripts/verify_smap_msl_results.sh` from a clean checkout and paste its output into this file.
- **Protocol binding:** the recorded run used `--expect-protocol-sha256 927cf78fe7acc1cdc2a044a800b1e7aef2d954cd7e3b60a5b6b5b31b552b8ea3`.

## Checksum manifest

```text
01c3d5422fe486f4bb771fa9c7eeba0cf0f578b4dad8119c3d02458741a881cc  smap_msl_comparisons.csv
80fbb59c6d3eba8b6200e130e2dc6aa50ce31a5f2d6da57c0055ae1138ccdeb0  smap_msl_event_f1.svg
ed62d393b9175a1921ea151924aefee5792f2fe3e8561d6b5c9178ebe4ee6bac  smap_msl_per_channel.csv
5263abb0359e1f19ba9da8a6f32e3c5c6e9c788809f929c1c066930facf30f72  smap_msl_results.json
ff5de8c39a1bc94791d56e512311fe6e4f49dc372e9e7195236069c531fca2ba  smap_msl_summary.csv
```

## Original run environment

Source: `reports/smap_msl_results/smap_msl_run_metadata.json`.

| Field | Value |
|---|---|
| Project | OES-Resilience |
| Package version | 0.4.0 |
| Python | 3.13.5 |
| NumPy | 2.5.3 |
| Platform | Linux-6.12.94+-x86_64-with-glibc2.41 |
| Run started | 2026-10-03T22:30:15.229471+00:00 |
| Elapsed time | 1.593465046986239 seconds |

## Original command

```text
smap-msl evaluate \
  --data-dir /workspace/oes-smap/cache/smap_msl \
  --protocol reports/smap_msl_protocol.json \
  --expect-protocol-sha256 927cf78fe7acc1cdc2a044a800b1e7aef2d954cd7e3b60a5b6b5b31b552b8ea3 \
  --out-dir reports/smap_msl_results \
  --verify
```

## Verification procedure

From the repository root:

```bash
bash scripts/verify_smap_msl_results.sh
```

The script performs:

1. A SHA-256 verification of every artifact listed in `reports/smap_msl_results/SHA256SUMS`.
2. A display of the recorded run metadata and protocol expectation.

## Acceptance rule

Step 1 is complete only when the script reports `OK` for all five files and its output is appended below with the verifying commit SHA, date, operator, and operating system.

## Local verification log

_To be appended after running the verifier on a clean checkout._
