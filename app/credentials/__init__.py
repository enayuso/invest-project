"""KIS credential verification.

Through 2026-08-06 this package also owned the per-user encrypted Postgres
credential store (`store.py`, `crypto.py`) — removed in the 2026-08-07
single-account rollback (docs/single-user-rollback-guide.md Phase 2).
KIS app key/secret/account number are `Settings.kis_credentials()` fields now
(app/config.py), sourced from env vars Jenkins injects at deploy time.

What remains — `validation.py` — is unchanged: proving a credential set
actually works by issuing a real KIS token. It used to gate the onboarding
screen's PUT /api/credentials; now app.web.state's lifespan calls it once at
boot instead (`_verify_configured_kis_credentials`), for the same reason —
a typo'd secret should surface as a boot-time log line, not as a wall of
"KIS call failed" warnings once the market opens.
"""
