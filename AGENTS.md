# Project memory

RiskSifter is a local video mining and review workbench. Core requirements are in
`RiskSifter_Codex_Project_Brief.md`. Current scope is Phase 1 only; no RL, RSI or
self-improving features.

## Engineering principles

- Prefer the simplest implementation that fully satisfies the requirements.
- Follow KISS and YAGNI. Avoid speculative features and premature abstractions.
- Do not add hashing, checksums, manifests, caching layers or custom logging
  unless explicitly required or justified by a concrete failure mode.
- Keep logging to important milestones, warnings and actionable errors. Avoid
  verbose per-item logs or printing full annotation payloads.
- Do not add redundant validation, broad exception handlers or silent fallbacks
  that hide bugs. Unexpected errors must remain visible.
- Reuse existing utilities and dependencies; do not create new frameworks.
- Keep changes focused. Prioritize correctness, readability and maintainability.
- Run tests appropriate to the change; do not rerun expensive inference when the
  model, prompt and preprocessing have not changed.

## Existing requirements that justify retained mechanisms

- Source content hashes deduplicate imports. Evidence hashes identify exactly
  what the model viewed. Keep these because the Brief requires provenance.
- Saved video/motion indexes and completed runs prevent repeated decoding and
  inference. Reuse the current files/SQLite; do not add another cache layer.
- Export metadata retains licenses, model/prompt versions and review history.
  Keep one metadata file and one integrity pass, not duplicate inventories.
- ZIP range reads avoid downloading 9GB public-data archives for a tiny sample.
- Human edits must survive retries; preserve the SQLite revision transaction.

## Asset and publication hygiene

- Commit source, tests, concise documentation, one representative output and
  the final evidence needed to support experimental claims.
- Keep weights, videos, virtualenvs, application workspaces, generated exports,
  screenshots, build products and caches out of Git.
- Write new experiment output under ignored `data/results/`; promote only
  reviewed final results to `reports/`. Do not commit duplicate JSON/text logs.
- Keep a usable local model and sample. Remove known failed downloads,
  duplicate exports and disposable build/cache files during requested cleanup;
  never delete user source videos or review history to save space.
- Automatic labels remain pending. Small object-mention checks are not accident
  detection accuracy. Preserve honest limitations and data attribution.

## Commands

`python -m pytest -q` runs offline tests. `risksifter serve` starts the loopback-only
workbench. `pip install -e '.[test]'` installs the core; `.[vlm]` is optional.
