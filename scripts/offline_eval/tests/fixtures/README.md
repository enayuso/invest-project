# Test fixtures

The bundle tests generate a minimal immutable-style fixture inside a temporary
directory. Keeping the fixture generated makes each hash relationship explicit
in the test and prevents stale binary/hash pairs from being checked in.

The Phase 3C session adapter also pins the nine approved synthetic behaviors
S01–S09 in `test_session_adapter.py`, including the known S08 false negative and
S09 false positive. These synthetic outcomes are never combined with real-claim
metrics.
