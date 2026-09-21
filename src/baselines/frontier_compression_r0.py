"""Freeze the frontier-compression cohorts and audit method eligibility/assets."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path


ROOT = Path("results/v2_rank_then_cut")
OUT = ROOT / "frontier_compression_r0"
CFG = Path("configs/frontier_compression_r0.json")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    cfg = json.loads(CFG.read_text())
    assert cfg["status"] == "FROZEN_BEFORE_SCREEN"
    source = ROOT / "v17canon_p0_fresh_v8_prefix_chain/per_query.jsonl"
    ids = sorted({r["example_id"] for r in read_jsonl(source)}, key=lambda x: (hashlib.sha256(x.encode()).hexdigest(), x))
    assert len(ids) == 1421
    screen = ids[: cfg["screen_queries"]]
    design = ids[cfg["screen_queries"] : cfg["screen_queries"] + cfg["design_queries"]]
    assert not set(screen) & set(design)
    model = Path(cfg["target_model"])
    required_model_files = [model / "config.json", model / "tokenizer_config.json"]
    for path in required_model_files:
        assert path.exists(), path
    compatibility = {
        "longllmlingua": {"eligible": True, "target_trainable": False, "interface": "input_ids", "asset_state": "official_package_installed"},
        "llmlingua2": {"eligible": True, "target_trainable": False, "interface": "input_ids", "asset_state": "official_package_installed_model_fetch_pending_or_cached"},
        "recomp": {"eligible": True, "target_trainable": False, "interface": "input_ids", "asset_state": "optional_domain_mismatched_control"},
        "xrag": {"eligible": True, "target_trainable": False, "interface": "inputs_embeds", "asset_state": "official_mistral_checkpoint_available_qwen_port_required"},
        "500xcompressor": {"eligible": True, "target_trainable": False, "interface": "layerwise_kv", "asset_state": "official_repository_but_public_weights_unavailable_qwen_port_required"},
        "sac": {"eligible": True, "target_trainable": False, "interface": "layerwise_kv", "asset_state": "independent_modified_encoder_qwen_port_required"},
        "flexcomp": {"eligible": True, "target_trainable": False, "interface": "inherits_backbone", "asset_state": "blocked_until_fixed_budget_backbone_passes"},
        "pisco": {"eligible": False, "reason": "paper trains decoder LoRA"},
        "ram": {"eligible": False, "reason": "decoder-freezing boundary not established for strict port"}
    }
    manifest = {
        "protocol": cfg["protocol"],
        "target_model": cfg["target_model"],
        "target_model_config_sha256": sha(model / "config.json"),
        "target_tokenizer_config_sha256": sha(model / "tokenizer_config.json"),
        "source_population_file": str(source),
        "source_population_sha256": sha(source),
        "population_count": len(ids),
        "screen_query_ids": screen,
        "design_query_ids": design,
        "selection": cfg["selection"],
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "vllm", "llmlingua")},
        "compatibility": compatibility,
        "sealed_sets_read": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "frozen R0 manifest changed"
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {
        "protocol": cfg["protocol"],
        "population_count": len(ids),
        "screen_count": len(screen),
        "design_count": len(design),
        "eligible_methods": [k for k, v in compatibility.items() if v.get("eligible")],
        "excluded_methods": {k: v["reason"] for k, v in compatibility.items() if not v.get("eligible")},
        "decision": "GO_R0_RUNTIME_IDENTITY_AND_HARD_SCREEN_ASSET_PREFLIGHT",
        "sealed_sets_read": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
