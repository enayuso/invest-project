"""Bounded, private, immutable evidence storage. 사설 증거 저장과 무결성 검사."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from pathlib import Path
from typing import Any

EVENT_LIMIT = 2 * 1024 * 1024
RUN_LIMIT = 16 * 1024 * 1024
SCHEMA = "rm-preparse-v1"
FILES = frozenset({"request.json", "response.json", "provenance.json", "invariants.json",
                   "inspection.json", "manifest.json", "error.json"})


class Refusal(Exception):
    """Only a fixed reason code may cross the CLI boundary. 이유 코드만 출력."""


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise Refusal("unsupported_json") from None


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_json(value: Any, *, limit: int = EVENT_LIMIT) -> bytes:
    from app.llm.request_capture import _contains_sensitive, _preflight

    try:
        _preflight(value, limit)
    except Exception:
        raise Refusal("event_limit_or_type") from None
    if _contains_sensitive(value):
        raise Refusal("rejected_sensitive")
    data = canonical(value)
    if len(data) > limit:
        raise Refusal("event_limit")
    return data


def _windows_acl(path: Path) -> None:
    """Check real DACLs, not POSIX mode bits. Windows 실제 DACL 확인."""
    import ctypes as c
    from ctypes import wintypes as w

    adv = c.WinDLL("advapi32", use_last_error=True)
    kernel = c.WinDLL("kernel32", use_last_error=True)
    adv.GetNamedSecurityInfoW.argtypes = [w.LPWSTR, w.DWORD, w.DWORD,
        c.POINTER(c.c_void_p), c.c_void_p, c.POINTER(c.c_void_p), c.c_void_p, c.POINTER(c.c_void_p)]
    adv.GetNamedSecurityInfoW.restype = w.DWORD
    adv.ConvertSidToStringSidW.argtypes = [c.c_void_p, c.POINTER(w.LPWSTR)]
    adv.ConvertSidToStringSidW.restype = w.BOOL
    adv.OpenProcessToken.argtypes = [w.HANDLE, w.DWORD, c.POINTER(w.HANDLE)]
    adv.OpenProcessToken.restype = w.BOOL
    adv.GetTokenInformation.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.POINTER(w.DWORD)]
    adv.GetTokenInformation.restype = w.BOOL
    adv.GetAce.argtypes = [c.c_void_p, w.DWORD, c.POINTER(c.c_void_p)]
    adv.GetAce.restype = w.BOOL
    kernel.GetCurrentProcess.restype = w.HANDLE
    kernel.LocalFree.argtypes = [c.c_void_p]
    kernel.CloseHandle.argtypes = [w.HANDLE]

    def sid_text(pointer):
        value = w.LPWSTR()
        if not adv.ConvertSidToStringSidW(pointer, c.byref(value)):
            raise Refusal("acl_unavailable")
        try:
            return value.value
        finally:
            kernel.LocalFree(c.cast(value, c.c_void_p))

    token = w.HANDLE()
    if not adv.OpenProcessToken(kernel.GetCurrentProcess(), 8, c.byref(token)):
        raise Refusal("acl_unavailable")
    try:
        needed = w.DWORD()
        adv.GetTokenInformation(token, 1, None, 0, c.byref(needed))
        buffer = c.create_string_buffer(needed.value)
        if not adv.GetTokenInformation(token, 1, buffer, needed, c.byref(needed)):
            raise Refusal("acl_unavailable")
        current = sid_text(c.cast(buffer, c.POINTER(c.c_void_p))[0])
    finally:
        kernel.CloseHandle(token)
    owner, dacl, descriptor = c.c_void_p(), c.c_void_p(), c.c_void_p()
    if adv.GetNamedSecurityInfoW(str(path), 1, 5, c.byref(owner), None,
                                 c.byref(dacl), None, c.byref(descriptor)):
        raise Refusal("acl_unavailable")
    try:
        if not dacl.value or sid_text(owner) != current:
            raise Refusal("not_private")
        # ACL header: revision/reserved/size/ace_count/reserved.
        # ACL 헤더의 ACE 개수를 읽고 모든 허용 주체를 검사한다.
        ace_count = c.c_ushort.from_address(dacl.value + 4).value
        # Python's Windows mkdir(0o700) grants OWNER RIGHTS to the verified owner.
        # Windows mkdir가 쓰는 OWNER RIGHTS는 위에서 확인한 현재 소유자에게만 적용된다.
        allowed = {current, "S-1-3-4", "S-1-5-18", "S-1-5-32-544"}
        for index in range(ace_count):
            ace = c.c_void_p()
            if not adv.GetAce(dacl, index, c.byref(ace)):
                raise Refusal("acl_unavailable")
            kind = c.c_ubyte.from_address(ace.value).value
            if kind == 1:  # ACCESS_DENIED_ACE cannot grant access.
                continue
            if kind != 0 or sid_text(ace.value + 8) not in allowed:
                raise Refusal("not_private")
    finally:
        kernel.LocalFree(descriptor)


def verify_private(path: Path, *, forbidden: tuple[Path, ...] = ()) -> Path:
    if not path.is_absolute() or str(path).startswith(("\\\\", "//")):
        raise Refusal("invalid_directory")
    try:
        for part in (path, *path.parents):
            info = part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise Refusal("reparse_directory")
        resolved = path.resolve(strict=True)
        if not path.is_dir() or path != resolved:
            raise Refusal("invalid_directory")
        if any(resolved == root.resolve() or root.resolve() in resolved.parents
               for root in forbidden):
            raise Refusal("source_directory")
        if any((part / ".git").exists() for part in (resolved, *resolved.parents)):
            raise Refusal("source_directory")
        synced = {"onedrive", "dropbox", "google drive", "googledrive", "iclouddrive", "box"}
        if any(part.lower().startswith("onedrive") or part.lower() in synced for part in path.parts):
            raise Refusal("shared_directory")
        for key in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
            if os.environ.get(key):
                shared = Path(os.environ[key]).resolve()
                if resolved == shared or shared in resolved.parents:
                    raise Refusal("shared_directory")
        if os.name == "nt":
            import ctypes
            if ctypes.windll.kernel32.GetDriveTypeW(str(resolved.anchor)) != 3:
                raise Refusal("nonlocal_directory")
            _windows_acl(resolved)
        elif resolved.stat().st_uid != os.getuid() or resolved.stat().st_mode & 0o077:
            raise Refusal("not_private")
        return resolved
    except OSError:
        raise Refusal("directory_unavailable") from None


class Evidence:
    """One immutable run, with a final manifest written last. 실행 단위 불변 증거."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.entries: dict[str, dict[str, Any]] = {}
        self.total = 0

    @classmethod
    def create(cls, root: Path, *, forbidden: tuple[Path, ...] = ()) -> "Evidence":
        root = verify_private(root, forbidden=forbidden)
        directory = root / str(uuid.uuid4())
        directory.mkdir(mode=0o700)
        verify_private(directory, forbidden=forbidden)
        return cls(directory)

    def write(self, name: str, value: Any) -> None:
        if name not in FILES or name in self.entries:
            raise Refusal("file_collision")
        verify_private(self.directory)
        data = checked_json(value)
        if self.total + len(data) > RUN_LIMIT:
            raise Refusal("run_limit")
        destination = self.directory / name
        temporary = self.directory / (name + ".partial")
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                if os.name == "nt":
                    _windows_acl(temporary)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            # Exclusive hard-link promotion never replaces an existing final file.
            # 최종 파일 독점 생성으로 기존 증거 덮어쓰기를 막는다.
            os.link(temporary, destination)
            temporary.unlink()
            if os.name != "nt":
                directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except OSError:
            raise Refusal("write_failed") from None
        self.entries[name] = {"bytes": len(data), "file_sha256": digest(data),
                              "canonical_sha256": digest(canonical(value))}
        self.total += len(data)

    def finish(self, **state: Any) -> dict[str, Any]:
        manifest = {"schema": SCHEMA, "run_id": self.directory.name, "files": dict(self.entries),
                    "pipeline_completed": False, "runtime_parse_observed": False,
                    "semantic_review": "not_reviewed", **state}
        self.write("manifest.json", manifest)
        verify_bundle(self.directory)
        return manifest


def read_json(path: Path) -> Any:
    try:
        if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
            raise Refusal("reparse_file")
        if path.stat().st_size > EVENT_LIMIT:
            raise Refusal("event_limit")
        if os.name == "nt":
            _windows_acl(path)
        elif path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
            raise Refusal("not_private")
        data = path.read_bytes()
        value = json.loads(data)
        if checked_json(value) != data:
            raise Refusal("noncanonical_evidence")
        return value
    except (OSError, ValueError):
        raise Refusal("evidence_unavailable") from None


def verify_bundle(directory: Path) -> dict[str, Any]:
    verify_private(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("run_id") != directory.name:
        raise Refusal("manifest_invalid")
    entries = manifest.get("files")
    if not isinstance(entries, dict) or not entries or set(entries) - (FILES - {"manifest.json"}):
        raise Refusal("manifest_invalid")
    if {p.name for p in directory.iterdir()} != set(entries) | {"manifest.json"}:
        raise Refusal("missing_or_orphan_evidence")
    total = (directory / "manifest.json").stat().st_size
    for name, expected in entries.items():
        value = read_json(directory / name)
        data = canonical(value)
        total += len(data)
        if expected != {"bytes": len(data), "file_sha256": digest(data), "canonical_sha256": digest(data)}:
            raise Refusal("evidence_hash_mismatch")
    if total > RUN_LIMIT:
        raise Refusal("run_limit")
    phase = manifest.get("phase")
    if (manifest.get("pipeline_completed") is not False or manifest.get("runtime_parse_observed") is not False
            or type(manifest.get("capture_complete")) is not bool):
        raise Refusal("manifest_invalid")
    required = {"request.json", "provenance.json"}
    if phase == "capture":
        required |= {"invariants.json"}
        if manifest.get("capture_complete"):
            required |= {"response.json"}
            if (manifest.get("sdk_calls_started"), manifest.get("sdk_calls_completed")) != (1, 1):
                raise Refusal("manifest_invalid")
            if manifest.get("invariants_status") != "PASS" or manifest.get("safety_trips") != 0:
                raise Refusal("manifest_invalid")
            inv = read_json(directory / "invariants.json")
            if inv.get("status") != "PASS" or inv.get("cleanup_ok") is not True or inv.get("safety_trips") != 0:
                raise Refusal("manifest_invalid")
            response = read_json(directory / "response.json")
            if digest(canonical(response.get("raw_response"))) != response.get("raw_response_canonical_sha256"):
                raise Refusal("evidence_hash_mismatch")
    elif phase == "inspect":
        required = {"inspection.json", "provenance.json"}
        if manifest.get("capture_complete"):
            raise Refusal("manifest_invalid")
    elif phase == "prepare":
        if (manifest.get("capture_complete") or manifest.get("status") != "prepared" or
                manifest.get("sdk_calls_started") != 0 or manifest.get("sdk_calls_completed") != 0):
            raise Refusal("manifest_invalid")
    else:
        raise Refusal("manifest_invalid")
    if not required <= set(entries):
        raise Refusal("missing_evidence")
    return manifest
