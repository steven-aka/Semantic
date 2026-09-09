from __future__ import annotations

import json
import hashlib
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: str | Path) -> str:
    """Hash a directory from sorted relative filenames and file digests."""
    root = Path(path)
    digest = hashlib.sha256()
    for file_path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(str(file_path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(file_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"


def gpu_names() -> list[str]:
    try:
        import torch
    except ImportError:
        return []
    return [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]


def experiment_metadata(**fields: Any) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu_names": gpu_names(),
        **fields,
    }


def write_metadata(path: str | Path, fields: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(fields), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
