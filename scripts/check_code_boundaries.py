"""Fail when baseline and proposed-method implementation boundaries are crossed."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "src/baselines"
METHOD_DIRS = tuple(ROOT / "src" / name for name in ("model", "training", "search", "representation"))
FORBIDDEN_FROM_BASELINES = ("src.model", "src.training", "src.search", "src.representation")


def imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
        elif isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
    return found


def main() -> None:
    violations: list[str] = []
    for path in BASELINES.glob("*.py"):
        for module in imports(path):
            if module.startswith(FORBIDDEN_FROM_BASELINES):
                violations.append(f"baseline {path.relative_to(ROOT)} imports method module {module}")
    for directory in METHOD_DIRS:
        for path in directory.rglob("*.py"):
            for module in imports(path):
                if module.startswith("src.baselines"):
                    violations.append(f"method {path.relative_to(ROOT)} imports baseline module {module}")
    if violations:
        raise SystemExit("\n".join(violations))
    print("code boundaries OK: baselines and proposed method are dependency-isolated")


if __name__ == "__main__":
    main()
