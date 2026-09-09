from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def gpu_status() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "reason": "nvidia-smi not found", "devices": []}
    result = subprocess.run(
        [executable, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    devices = [line for line in lines if "," in line]
    diagnostics = [line for line in lines if "," not in line]
    reason = result.stderr.strip() or "\n".join(diagnostics) or None
    return {
        # A single broken NVML handle can make nvidia-smi exit non-zero while
        # still returning valid rows for the healthy GPUs on shared servers.
        "available": bool(devices),
        "reason": reason,
        "devices": devices,
    }


def cached_model(cache_root: Path, model_name: str, local_directory: Path | None = None) -> dict[str, Any]:
    directory = cache_root / "hub" / ("models--" + model_name.replace("/", "--"))
    snapshots = directory / "snapshots"
    weight_files = list(snapshots.rglob("*.safetensors")) if snapshots.exists() else []
    tokenizer_files = list(snapshots.rglob("tokenizer.json")) if snapshots.exists() else []
    local_weights = list(local_directory.glob("*.safetensors")) if local_directory and local_directory.exists() else []
    local_partials = list(local_directory.glob("*.part")) if local_directory and local_directory.exists() else []
    local_tokenizer = bool(local_directory and (local_directory / "tokenizer.json").exists())
    expected_local_weights = 0
    local_index = local_directory / "model.safetensors.index.json" if local_directory else None
    if local_index and local_index.exists():
        with local_index.open(encoding="utf-8") as handle:
            expected_local_weights = len(set(json.load(handle)["weight_map"].values()))
    local_complete = (
        expected_local_weights > 0
        and len(local_weights) == expected_local_weights
        and not local_partials
    )
    return {
        "model": model_name,
        "cache_directory_exists": directory.exists(),
        "local_directory": str(local_directory) if local_directory else None,
        "tokenizer_cached": bool(tokenizer_files) or local_tokenizer,
        "weight_file_count": len(weight_files) + len(local_weights),
        "expected_local_weight_file_count": expected_local_weights,
        "partial_file_count": len(local_partials),
        "weights_cached": bool(weight_files) or local_complete,
    }


def build_report(cache_root: Path) -> dict[str, Any]:
    packages = {
        name: package_version(name)
        for name in ("torch", "transformers", "datasets", "vllm", "matplotlib", "pyyaml")
    }
    gpu = gpu_status()
    target = cached_model(cache_root, "Qwen/Qwen3-8B", Path("models/Qwen3-8B"))
    teacher = cached_model(cache_root, "Qwen/Qwen3-14B", Path("models/Qwen3-14B"))
    data = {
        "raw_hotpot": Path("data/raw/hotpot_validation.parquet").exists(),
        "smoke_examples": Path("data/units/hotpot_exact_pilot.jsonl").exists(),
        "controlled_examples": Path("data/units/hotpot_controlled_pilot.jsonl").exists(),
    }
    return {
        "python": platform.python_version(),
        "packages": packages,
        "gpu": gpu,
        "target": target,
        "teacher": teacher,
        "data": data,
        "ready": {
            "data_preparation": all(data.values()) and target["tokenizer_cached"],
            "full_context": gpu["available"] and packages["vllm"] is not None and target["weights_cached"],
            "packet_generation": gpu["available"] and packages["vllm"] is not None and teacher["weights_cached"],
            "exact_search": gpu["available"] and packages["vllm"] is not None and target["weights_cached"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check readiness of each V0 experiment stage")
    parser.add_argument("--cache-root", default=".cache/huggingface")
    parser.add_argument("--output", default="results/preflight.json")
    args = parser.parse_args()
    report = build_report(Path(args.cache_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report["ready"], ensure_ascii=False))


if __name__ == "__main__":
    main()
