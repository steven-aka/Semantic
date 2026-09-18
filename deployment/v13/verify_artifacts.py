from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(Path("deployment/v13/artifact_manifest.json").read_text())
    failures = []
    for index, item in enumerate(manifest["files"], 1):
        path = Path(item["path"])
        if not path.is_file():
            failures.append(f"missing: {path}")
        elif path.stat().st_size != item["size"]:
            failures.append(f"size: {path}")
        elif sha256(path) != item["sha256"]:
            failures.append(f"sha256: {path}")
        if index % 500 == 0:
            print(f"verified {index}/{manifest['total_files']}", flush=True)
    if failures:
        raise SystemExit("artifact verification failed:\n" + "\n".join(failures[:20]))
    print(
        f"verified {manifest['total_files']} files "
        f"({manifest['total_bytes']} bytes)",
        flush=True,
    )


if __name__ == "__main__":
    main()
