"""Local Docker stub checks deploy preflight without touching a container."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def shell():
    if os.name == "nt":
        candidate = Path(r"C:\Program Files\Git\bin\bash.exe")
        if candidate.is_file():
            return str(candidate)
    candidate = shutil.which("bash")
    if candidate is None:
        pytest.skip("bash unavailable")
    return candidate


def _run(tmp_path: Path, shell: str, env_body: str | bytes, *, approved_mount: bool = False):
    root = Path(__file__).resolve().parents[2]
    (tmp_path / ".env.dev").write_bytes(
        env_body if isinstance(env_body, bytes) else env_body.encode("utf-8"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$DOCKER_CALLS"\n'
                      'if [ "$1" = image ]; then printf "\\n"; fi\n', encoding="utf-8")
    docker.chmod(0o755)
    calls = tmp_path / "docker-calls"
    environment = os.environ.copy()
    environment.update(WORKSPACE=str(tmp_path), DOCKER_CALLS=str(calls),
                       PATH=str(bin_dir) + os.pathsep + environment["PATH"])
    if approved_mount:
        host = "/var/lib/invest-eval/rm-capture/synthetic"
        bash_env = tmp_path / "capture-stub.bash"
        bash_env.write_text(
            'function [ {\n'
            f'  if [[ "$1" == "!" && "$2" == "-d" && "$3" == "{host}" ]]; then return 1; fi\n'
            '  builtin [ "$@"\n'
            '}\n'
            f'readlink() {{ printf "%s\\n" "{host}"; }}\n'
            'stat() { if [ "$2" = "%a" ]; then echo 700; else echo 0; fi; }\n'
            'findmnt() { echo /var/lib/invest-eval/rm-capture; }\n'
            'df() { if [ "$2" = "--output=size" ]; then printf "size\\n1073741824\\n"; '
            'else printf "avail\\n800000000\\n"; fi; }\n'
            'du() { printf "1000000\\t/var/lib/invest-eval/rm-capture\\n"; }\n',
            encoding="utf-8")
        environment["BASH_ENV"] = bash_env.as_posix()
    completed = subprocess.run([shell, str(root / "pipeline.sh"), "dev"],
                               cwd=tmp_path, env=environment, text=True,
                               encoding="utf-8", errors="replace",
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=15, check=False)
    return completed, calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []


@pytest.mark.parametrize("setting", [
    "RM_REQUEST_CAPTURE_ENABLED=maybe\n",
    "RM_REQUEST_CAPTURE_ENABLED=true\nRM_REQUEST_CAPTURE_DIRECTORY=/tmp/wrong\n",
    "RM_REQUEST_CAPTURE_ENABLED=true\nRM_REQUEST_CAPTURE_DIRECTORY=/var/lib/invest-eval/rm-capture/current\n",
    "RM_REQUEST_CAPTURE_ENABLED=false\nRM_REQUEST_CAPTURE_ENABLED=true\n",
    "RM_REQUEST_CAPTURE_ENABLED=\n",
    "rm_request_capture_enabled=true\nrm_request_capture_directory=/tmp\n",
    " RM_REQUEST_CAPTURE_ENABLED=true\n",
    "export RM_REQUEST_CAPTURE_ENABLED=true\n",
    "RM_REQUEST_CAPTURE_ENABLED=false\nRM_REQUEST_CAPTURE_ENABLED=maybe\n",
    "RM_REQUEST_CAPTURE_ENABLED=false\nRM_REQUEST_CAPTURE_EVENT_MAX_BYTES=bad\n",
    "RM_REQUEST_CAPTURE_ENABLED=true\nRM_REQUEST_CAPTURE_EVENT_MAX_BYTES=1\n",
    "RM_REQUEST_CAPTURE_UNKNOWN=true\n",
    "\ufeffRM_REQUEST_CAPTURE_ENABLED=true\nRM_REQUEST_CAPTURE_DIRECTORY=/tmp\n",
    "\ufeffrm_request_capture_enabled=true\nRM_REQUEST_CAPTURE_ENABLED=false\n",
    "\ufeffRM_REQUEST_CAPTURE_ENABLED=false\n",
    b"RM_REQUEST_CAPTURE_ENABLED=false\n# invalid \xff\n",
])
def test_invalid_capture_settings_never_call_docker(tmp_path, shell, setting):
    completed, calls = _run(tmp_path, shell, setting)
    assert completed.returncode != 0
    assert calls == []


@pytest.mark.parametrize("space", (
    "\u0085", "\u00a0", "\u1680", *(chr(code) for code in range(0x2000, 0x200B)),
    "\u2028", "\u2029", "\u202f", "\u205f", "\u3000", " \t\u2003",
))
def test_unicode_leading_space_never_bypasses_capture_preflight(tmp_path, shell, space):
    completed, calls = _run(tmp_path, shell, f"{space}RM_REQUEST_CAPTURE_ENABLED=true\n"
                             "RM_REQUEST_CAPTURE_DIRECTORY=/tmp\n")
    assert completed.returncode != 0
    assert calls == []


@pytest.mark.parametrize("setting", [
    "RM_REQUEST_CAPTURE_ENABLED=false\n\u3000RM_REQUEST_CAPTURE_ENABLED=true\n",
    "RM_REQUEST_CAPTURE_ENABLED=false\n\u2003RM_REQUEST_CAPTURE_DIRECTORY=/tmp\n",
    "RM_REQUEST_CAPTURE_ENABLED=false\n\u00a0RM_REQUEST_CAPTURE_EVENT_MAX_BYTES=1\n",
    " \t\u0085rm_request_capture_enabled=true\n",
])
def test_unicode_alias_or_duplicate_never_calls_docker(tmp_path, shell, setting):
    completed, calls = _run(tmp_path, shell, setting)
    assert completed.returncode != 0
    assert calls == []


def test_capture_off_keeps_existing_deploy_call_shape(tmp_path, shell):
    completed, calls = _run(tmp_path, shell, "RM_REQUEST_CAPTURE_ENABLED=false\n")
    assert completed.returncode == 0
    assert [call.split()[0] for call in calls] == ["build", "stop", "rm", "run"]
    assert "--env-file" in calls[-1]
    assert "invest-project-kis-token:/var/cache/kis" in calls[-1]
    assert "KIS_TOKEN_CACHE_DIR=/var/cache/kis" in calls[-1]
    assert "/var/lib/invest-eval/rm-capture/current" not in calls[-1]


def test_capture_on_passes_exact_approved_mount_to_docker_stub(tmp_path, shell):
    host = "/var/lib/invest-eval/rm-capture/synthetic"
    env = ("RM_REQUEST_CAPTURE_ENABLED=true\n"
           "RM_REQUEST_CAPTURE_DIRECTORY=/var/lib/invest-eval/rm-capture/current\n"
           f"RM_REQUEST_CAPTURE_HOST_DIRECTORY={host}\n")
    completed, calls = _run(tmp_path, shell, env, approved_mount=True)
    assert completed.returncode == 0, completed.stderr
    assert [call.split()[0] for call in calls] == ["build", "image", "stop", "rm", "run"]
    assert f"--mount type=bind,src={host},dst=/var/lib/invest-eval/rm-capture/current" in calls[-1]
    assert "invest-project-kis-token:/var/cache/kis" in calls[-1]
