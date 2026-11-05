# SMAP/MSL replication archive instructions

## Purpose

This file defines the externally checkable replication package for the locked SMAP/MSL evaluation. The archive must contain the exact protocol, code state, result artifacts, checksums, environment, and commands needed to reproduce or audit the reported result.

## Archive status

- **Final report:** complete.
- **Primary result:** criterion not met.
- **Zenodo DOI:** **PENDING**. Create the versioned Zenodo deposit after the final archive files are committed, then record the DOI in this file, `CITATION.cff`, `README.md`, and `REVIEW.md`.

## Archive contents

Include these repository paths in the Zenodo deposit:

```text
CITATION.cff
LICENSE
README.md
REVIEW.md
REVIEW_GUIDE.md
CHANGELOG.md
pyproject.toml
reports/smap_msl_protocol.json
reports/smap_msl_protocol.md
reports/smap_msl_data.sha256
reports/smap_msl_results/SHA256SUMS
reports/smap_msl_results/smap_msl_comparisons.csv
reports/smap_msl_results/smap_msl_event_f1.svg
reports/smap_msl_results/smap_msl_per_channel.csv
reports/smap_msl_results/smap_msl_results.json
reports/smap_msl_results/smap_msl_run_metadata.json
reports/smap_msl_results/smap_msl_summary.csv
reports/smap_msl_results/VERIFICATION.md
reports/smap_msl_results/BASELINE_COMPARISONS.md
reports/smap_msl_results/FINAL_REPORT.md
reports/smap_msl_results/REPLICATION.md
reports/smap_msl_ablations/ABLATION_PLAN.md
scripts/verify_smap_msl_results.sh
```

Also include the source tree needed to run the evaluation, excluding datasets, caches, virtual environments, and other generated artifacts.

## Code state

Record the exact commit used for the archive. The evaluation-report lineage includes:

- `7d7487e`: verification manifest and environment record.
- `d802176`: final baseline comparisons and confidence intervals.
- `f7538b6`: ablation preregistration status.

The Zenodo deposit must identify one final commit SHA and use that same commit for the archived source tree.

## Environment

The original run used:

- OES-Resilience 0.4.0
- Python 3.13.5
- NumPy 2.5.3
- Linux x86_64, glibc 2.41
- Run started 2026-10-03T22:30:15Z

The archive should include a lockfile or exact environment specification sufficient to recreate these versions.

## Verification procedure

From the repository root:

```bash
bash scripts/verify_smap_msl_results.sh
```

This verifies the SHA-256 checksums of:

- `smap_msl_comparisons.csv`
- `smap_msl_event_f1.svg`
- `smap_msl_per_channel.csv`
- `smap_msl_results.json`
- `smap_msl_summary.csv`

Dataset integrity is checked with:

```bash
sha256sum -c reports/smap_msl_data.sha256
```

The dataset itself is not redistributed in the repository or archive; obtain it from the sources recorded in the locked protocol and verify it against `reports/smap_msl_data.sha256`.

## Reproduction command

The original recorded command was:

```bash
smap-msl evaluate \
  --data-dir /workspace/oes-smap/cache/smap_msl \
  --protocol reports/smap_msl_protocol.json \
  --expect-protocol-sha256 927cf78fe7acc1cdc2a044a800b1e7aef2d954cd7e3b60a5b6b5b31b552b8ea3 \
  --out-dir reports/smap_msl_results \
  --verify
```

Replace `/workspace/oes-smap/cache/smap_msl` with the local path to the verified dataset.

## Zenodo release procedure

1. Create a Git tag such as `smap-msl-locked-evaluation-v1` at the final commit.
2. Create a GitHub release from that tag.
3. Create a Zenodo deposit and upload the archive contents listed above.
4. Use the repository metadata in `.zenodo.json` and cite the locked protocol and final report.
5. Publish the deposit to obtain a DOI.
6. Record the DOI below and update `CITATION.cff`, `README.md`, and `REVIEW.md`.

## DOI

**Pending.**
