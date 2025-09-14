# Contributing to wormWatcher (HWVA)
## Honesty Stack
1) Evidence > vibes: include paths, parameters, and seeds in commits/PRs.
2) Provenance: note input video names (no PII), config and code versions.
3) Clarify: if FPS, ROIs, or events timing are unknown, ASK before analysis.
4) Default-safe: do not infer stimuli timing unless verifiable.

## Workflow
- Branch from `main`, name: `feat/...`, `fix/...`, or `docs/...`.
- Run `make ci` locally before pushing.
- PR template must include: what changed, how verified, sample output/log snippet.
- Large media lives in `project_root/videos/` (LFS) only. Never commit raw data elsewhere.

## Coding
- Python ≥3.10; keep functions pure where possible.
- Log versions + seed; write outputs under `project_root/out/` with deterministic names.
- Add/adjust parameters in `project_root/config.yaml` rather than hardcoding.

## Tests/QC
- Add quick checks in `project_root/src/sanity.py` or minimal `pytest` tests for new modules.
