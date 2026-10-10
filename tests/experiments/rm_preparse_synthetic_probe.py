"""Fresh-process prepare acceptance with synthetic data. 신규 프로세스 무전송 수용 시험."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.experiments.replay import contract, pinned
from app.experiments.rm_diagnostics.evidence import canonical, verify_bundle
from replay_fixtures import make_row
from test_rm_preparse_capture import protect_windows


def main():
    from datetime import date
    import asyncpg

    # Count real DB entry points; any call also raises, so prepare cannot silently succeed with DB use.
    # 실제 DB 진입점 호출 횟수를 센다(호출 시 예외 → 조용한 통과 불가).
    db_calls = {"count": 0}

    def counted(*args, **kwargs):
        db_calls["count"] += 1
        raise AssertionError("DB access attempted in synthetic prepare")

    asyncpg.connect = counted
    asyncpg.create_pool = counted
    from app.experiments import rm_preparse_capture as cli

    os.environ["DATABASE_URL"] = "postgresql://synthetic.invalid/diagnostic"
    with tempfile.TemporaryDirectory(prefix="rm-synthetic-") as location:
        root = Path(location)
        if os.name == "nt":
            protect_windows(root)
        else:
            root.chmod(0o700)
        casesets = root / "casesets"
        row = make_row(id=1141, ticker="035420.KS", trade_date=date(2026, 9, 1))
        record = contract.record_from_row(row)
        contract.write_caseset(casesets, "pilot12", [record], {"synthetic": True})
        settings_file = root / "pinned.json"
        settings_file.write_bytes(canonical(pinned.default_document()))
        result = cli.main(["prepare", "--output-root", str(root), "--pinned", str(settings_file),
                           "--caseset-dir", str(casesets)])
        assert result == 0, "fresh process preparation failed"
        runs = [path for path in root.iterdir() if path.is_dir() and path.name != "casesets"]
        assert len(runs) == 1
        manifest = verify_bundle(runs[0])
        assert manifest["sdk_calls_started"] == manifest["sdk_calls_completed"] == 0
        assert not (runs[0] / "response.json").exists()
        assert db_calls["count"] == 0, "synthetic prepare touched the DB"
        print(json.dumps({"synthetic_prepare": "PASS", "sdk_calls": manifest["sdk_calls_started"],
                          "db_calls": db_calls["count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
