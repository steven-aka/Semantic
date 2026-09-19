"""Resume the preregistered B2B lineage after the already-running V8 finishes."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.data.schemas import read_jsonl
from src.reproducibility import sha256, write_metadata


ROOT = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
CONFIG = Path("configs/v17sel_b2b_lineage_clean_pool_gate.json")
PYTHON = ".venv/bin/python"


def status(stage: str, state: str, **extra: object) -> None:
    write_metadata(ROOT / "pipeline_status.json", {
        "stage": stage, "state": state, "updated_utc": datetime.now(timezone.utc).isoformat(), **extra,
    })


def run(stage: str, argv: list[str], env: dict[str, str]) -> None:
    log = ROOT / f"{stage}.log"
    status(stage, "running", command=argv, log=str(log))
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\nBEGIN {datetime.now(timezone.utc).isoformat()} {argv!r}\n")
        handle.flush()
        result = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, env=env, check=False)
    if result.returncode:
        status(stage, "failed", exit_code=result.returncode, log=str(log))
        raise RuntimeError(f"{stage} failed with exit {result.returncode}; see {log}")
    status(stage, "complete", log=str(log))


def write_config(name: str, payload: dict) -> Path:
    path = ROOT / f"{name}_protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"refusing to change frozen stage config: {path}")
    else:
        write_metadata(path, payload)
    return path


def publish_results(audit: Path) -> None:
    """Publish only small review artifacts; bulky checkpoints and JSONL remain local."""
    git = ["git", "--git-dir=.git-sync", "--work-tree=."]
    artifacts = [
        ROOT / "manifest.json", ROOT / "v8_clean_selected_checkpoint.json",
        ROOT / "v10_protocol.json", ROOT / "v12_protocol.json",
        ROOT / "v13_protocol.json", audit / "summary.json",
    ]
    subprocess.run([*git, "add", "--", *(str(path) for path in artifacts)], check=True)
    changed = subprocess.run([*git, "diff", "--cached", "--quiet", "--", *(str(path) for path in artifacts)], check=False)
    if changed.returncode == 1:
        subprocess.run([*git, "commit", "-m", "Record lineage-clean SEL-B2B pool coverage result", "--", *(str(path) for path in artifacts)], check=True)
    elif changed.returncode != 0:
        raise RuntimeError("could not inspect staged B2B artifacts")
    subprocess.run([*git, "push", "origin", "main"], check=True)


def check_manifest() -> dict:
    config = json.loads(CONFIG.read_text())
    if config["status"] != "FROZEN_APPROVED_TO_BUILD_AND_RUN":
        raise ValueError("B2B protocol is not approved")
    manifest = json.loads((ROOT / "manifest.json").read_text())
    if manifest["protocol_sha256"] != sha256(CONFIG):
        raise ValueError("manifest/protocol hash mismatch")
    for roles in manifest["files"].values():
        for item in roles.values():
            if sha256(item["output"]) != item["output_sha256"]:
                raise ValueError(f"split file changed: {item['output']}")
    if manifest["holdout_count"] != 581 or manifest["inner_validation_count"] != 550:
        raise ValueError("unexpected B2B split sizes")
    holdout = set(manifest["holdout_ids"])
    inner = set(manifest["inner_validation_ids"])
    if holdout & inner:
        raise ValueError("holdout and validation overlap")
    for roles in manifest["files"].values():
        train = {row["example_id"] for row in read_jsonl(roles["train"]["output"])}
        if train & (holdout | inner):
            raise ValueError("lineage contamination")
    return manifest


def wait_v8() -> None:
    pid_path = ROOT / "v8.pid"
    pid = int(pid_path.read_text().strip())
    while not (ROOT / "v8_run/training_metadata.json").is_file():
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        except FileNotFoundError as exc:
            raise RuntimeError("V8 stopped before writing training metadata") from exc
        if b"train_v8_sequential_policy" not in cmd or b"v17sel_b2b_lineage_clean_holdout" not in cmd:
            raise RuntimeError("V8 PID no longer matches this lineage run")
        status("v8", "waiting", pid=pid)
        time.sleep(20)
    if not (ROOT / "v8_run/selected_checkpoint.json").is_file():
        raise RuntimeError("V8 metadata exists but checkpoint selection is missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = check_manifest()
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        wait_v8()
        selected_file = ROOT / "v8_clean_selected_checkpoint.json"
        if not selected_file.exists():
            run("v8_select", [PYTHON, "-m", "src.evaluation.v17sel_b2b_select_v8_checkpoint"], env)
        selected = json.loads(selected_file.read_text())
        checkpoint = selected["selected"]["checkpoint"]
        if not Path(checkpoint, "training_metadata.json").is_file():
            raise RuntimeError("selected V8 checkpoint is incomplete")
        data = ROOT / "data"
        train = data / "v10_v12_train_train_clean.jsonl"
        valid = data / "v10_v12_train_inner_validation.jsonl"
        holdout = data / "v10_v12_train_holdout.jsonl"
        train_cache = ROOT / "v8_train_embeddings.pt"
        valid_cache = ROOT / "v8_inner_embeddings.pt"
        for name, source, cache in (("v8_train_cache", train, train_cache), ("v8_inner_cache", valid, valid_cache)):
            if not Path(str(cache) + ".manifest.json").is_file():
                run(name, [PYTHON, "-u", "-m", "src.data.cache_v10_encoder_embeddings", "--data", str(source), "--checkpoint", checkpoint, "--output", str(cache), "--manifest", str(cache) + ".manifest.json", "--batch-size", "16"], env)
        v10_out = ROOT / "v10_run"
        v10 = json.loads(Path("configs/v10_mask_value_probe.json").read_text())
        v10["protocol"] = "v17sel_b2b_lineage_clean_v10"
        v10["scientific_role"] = "train-only head initialization; no development or holdout evaluation"
        v10["initialization"].update(checkpoint=checkpoint, checkpoint_metadata_sha256=sha256(Path(checkpoint) / "training_metadata.json"), sequential_head_sha256=sha256(Path(checkpoint) / "sequential_head.pt"))
        v10["data"].update(train=str(train), train_sha256=sha256(train), train_examples=2032, train_cache=str(train_cache), train_cache_sha256=sha256(train_cache), development=str(valid), development_sha256=sha256(valid), development_examples=550, development_cache=str(valid_cache), development_cache_sha256=sha256(valid_cache))
        v10["output"] = str(v10_out)
        v10_config = write_config("v10", v10)
        if not (v10_out / "probe_metadata.json").is_file():
            run("v10", [PYTHON, "-u", "-m", "src.training.train_v10_mask_value_probe", "--protocol-config", str(v10_config), "--train-data", str(train), "--validation-data", str(valid), "--exact-dir", "results/v2_rank_then_cut/candidates5000_exact", "--checkpoint", checkpoint, "--train-cache", str(train_cache), "--validation-cache", str(valid_cache), "--output-dir", str(v10_out), "--max-optimizer-steps", "400", "--batch-size", "16", "--learning-rate", "0.0002", "--weight-decay", "0.01", "--warmup-steps", "40", "--mask-layers", "2", "--mask-heads", "8", "--dropout", "0.1", "--evaluation-mask-chunk-size", "256", "--seed", "20260912", "--skip-evaluation"], env)
        v12_out = ROOT / "v12_run"
        v12 = json.loads(Path("configs/v12_high_fidelity_retriever.json").read_text())
        v12["protocol"] = "v17sel_b2b_lineage_clean_v12"
        v12["scientific_role"] = "train-only V13 initialization; no development or holdout evaluation"
        v12["frozen"].update(v10_head=str(v10_out / "mask_value_head.pt"), v10_head_sha256=sha256(v10_out / "mask_value_head.pt"))
        v12["data"].update(train=str(train), train_sha256=sha256(train), train_cache=str(train_cache), train_cache_sha256=sha256(train_cache))
        v12["output"] = str(v12_out)
        v12_config = write_config("v12", v12)
        if not (v12_out / "metadata.json").is_file():
            run("v12", [PYTHON, "-u", "-m", "src.training.train_v12_high_fidelity_retriever", "--protocol-config", str(v12_config), "--train-data", str(train), "--train-cache", str(train_cache), "--v10-head", str(v10_out / "mask_value_head.pt"), "--output-dir", str(v12_out), "--steps", "400", "--batch-size", "16", "--lr", "0.0001", "--seed", "20260918"], env)
        v13_out = ROOT / "v13_run"
        v13 = json.loads(Path("configs/v13_joint_retriever.json").read_text())
        v13["protocol"] = "v17sel_b2b_lineage_clean_v13"
        v13["scientific_role"] = "lineage-clean train/inner-validation checkpoint selection; holdout sealed"
        v13["arguments"].update(train_data=str(data / "v13_train_train_clean.jsonl"), validation_data=str(data / "v13_train_inner_validation.jsonl"), checkpoint=checkpoint, initial_head=str(v12_out / "mask_retrieval_head.pt"), output_dir=str(v13_out))
        v13_config = write_config("v13", v13)
        if not (v13_out / "training_metadata.json").is_file():
            run("v13", [PYTHON, "-u", "-m", "src.training.train_v13_joint_retriever", "--protocol-config", str(v13_config), "--train-data", v13["arguments"]["train_data"], "--validation-data", v13["arguments"]["validation_data"], "--checkpoint", checkpoint, "--initial-head", v13["arguments"]["initial_head"], "--output-dir", str(v13_out), "--steps", "300", "--validation-interval", "100", "--batch-size", "16", "--gradient-accumulation", "1", "--lora-lr", "0.00001", "--head-lr", "0.0001", "--seed", "20260918"], env)
        v13_selected = json.loads((v13_out / "selected_checkpoint.json").read_text())["checkpoint"]
        if not Path(v13_selected, "mask_retrieval_head.pt").is_file():
            raise RuntimeError("V13 selected checkpoint incomplete")
        # The holdout is read only after V8, V10, V12 and V13 are frozen.
        holdout_cache = ROOT / "v13_holdout_embeddings.pt"
        if not Path(str(holdout_cache) + ".manifest.json").is_file():
            run("v13_holdout_cache", [PYTHON, "-u", "-m", "src.data.cache_v10_encoder_embeddings", "--data", str(holdout), "--checkpoint", v13_selected, "--output", str(holdout_cache), "--manifest", str(holdout_cache) + ".manifest.json", "--batch-size", "16"], env)
        rollouts = ROOT / "v8_holdout_rollouts.jsonl"
        if not Path(str(rollouts) + ".metadata.json").is_file():
            run("v8_holdout_rollout", [PYTHON, "-u", "-m", "src.evaluation.rollout_v8_policy", "--data", str(data / "v8_train_holdout.jsonl"), "--checkpoint", checkpoint, "--output", str(rollouts), "--batch-size", "8", "--beam-width", "8"], env)
        audit = ROOT / "pool_audit"
        if not (audit / "summary.json").is_file():
            run("topk_pool_audit", [PYTHON, "-u", "-m", "src.evaluation.v17sel_b2b_clean_pool_audit", "--data", str(holdout), "--cache", str(holdout_cache), "--head", str(Path(v13_selected) / "mask_retrieval_head.pt"), "--rollouts", str(rollouts), "--exact-dir", "results/v2_rank_then_cut/candidates5000_exact", "--config", str(CONFIG), "--output-dir", str(audit)], env)
        result = json.loads((audit / "summary.json").read_text())
        status("pipeline", "complete", decision=result["decision"], counts=result["oracle_pool_090_by_top_k"])
        try:
            publish_results(audit)
            status("pipeline", "complete_published", decision=result["decision"], counts=result["oracle_pool_090_by_top_k"])
        except Exception as exc:
            status("pipeline", "complete_publication_failed", decision=result["decision"], error=str(exc))
            raise
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        previous = json.loads((ROOT / "pipeline_status.json").read_text()) if (ROOT / "pipeline_status.json").is_file() else {}
        if previous.get("state") != "complete_publication_failed":
            status("pipeline", "failed", error=str(exc))
        raise
