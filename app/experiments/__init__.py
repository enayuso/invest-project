"""Standalone research tooling that calls `app.orchestrator.pipeline.
run_pipeline` directly instead of `app.backtest.harness`/`portfolio_harness`
— see `app.experiments.language_ab`'s module docstring for why a language
A/B comparison specifically needs that independence (no gap-stop replay, no
`reflect()`, no equity curve — all of it would be an extra language-
dependent confound, not signal).
"""
