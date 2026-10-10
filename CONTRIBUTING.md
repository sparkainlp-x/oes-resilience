# Contributing to OES-Resilience

Thank you for helping! Please read the [Code of Conduct](CODE_OF_CONDUCT.md) first. Report security issues privately as described in [SECURITY.md](SECURITY.md), not as public issues.

## Ground rules

- **Synthetic data only in this repository.** Never commit private, proprietary, personal or customer data, including telemetry "just for a test". Adapters for public datasets (for example the planned NASA SMAP/MSL adapters) must download data from its public source rather than vendoring it, and must respect its license.
- **Keep synthetic results and physical claims separate.** Numbers produced by this benchmark describe software behaviour on synthetic regimes. Do not present them as evidence about a physical sensor, spacecraft, quantum device, navigation (GPS) system or any production system. Wording in the code, docs and pull requests should make clear which is which.
- **Real measured numbers only.** Any number in the README or docs must come from a command in this repository that anyone can rerun (state the command and seed). Report regressions and losses as honestly as wins.
- **License.** The project is AGPL-3.0-only (see [LICENSE](LICENSE) and [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)). By contributing, you agree that your contribution is licensed under the same terms.

## Development setup

```bash
python -m pip install -e ".[test]"   # add ,iforest for the optional Isolation Forest detector
python -m pip install ruff
```

## Before you open a pull request

1. **Tests:** `python -m pytest --cov` must pass, with total coverage of at least 90% (CI enforces this on Python 3.10–3.13). Add tests for new behaviour.
2. **Lint:** `ruff check .` must be clean.
3. **Determinism:** if you change scoring, generation or calibration, rerun the example commands in `.github/workflows/ci.yml`. If an example CSV under `examples/` changes on purpose, regenerate it and explain why in the PR. The `--verify` flags of `compare` and `stress` must still report a hash match.
4. **Docs:** update the README, `docs/` and `CHANGELOG.md` (under *Unreleased*) for any user-visible change.
5. **New detectors:** follow [docs/detector-api.md](docs/detector-api.md). A detector must declare `supports_missing = True` and document its missing-data handling before it may receive NaN; silent zero-filling is not accepted.

Commit messages should say what changed and why. Keep pull requests focused.
