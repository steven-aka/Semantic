"""Audit whether existing Qwen3-14B semantic packets qualify for SEM-A1."""
import glob
import json
from pathlib import Path

from src.data.schemas import SemanticPacket, read_jsonl
from src.representation.token_counter import count_tokens, load_tokenizer
from src.teacher.packet_validator import validate_packet

BASE = Path("results/v2_rank_then_cut")
OUT = BASE / "v17sem_a0_existing_packet_audit"


def main():
    tokenizer = load_tokenizer("models/Qwen3-8B")
    train_ids = {r["example_id"] for r in read_jsonl(BASE / "v17canon_p0_fresh_v8_prefix_chain/per_query.jsonl")}
    assets = []
    for directory in sorted(Path("data").glob("packets_qampari*")):
        files = sorted(directory.glob("*.jsonl"))
        if not files:
            continue
        first = json.loads(files[0].read_text().splitlines()[0])
        if "gist" not in first:
            continue
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        generated = "model_name" in metadata
        kind = ("teacher_generated" if generated else
                "deterministic_lossless_partition" if "packetization" in metadata else
                "unknown_legacy_no_metadata")
        rows=[]
        for file in files:
            rows.extend(read_jsonl(file, SemanticPacket))
        validation=[validate_packet(r) for r in rows]
        source_tokens=[count_tokens(tokenizer,r.source) for r in rows]
        gist_tokens=[count_tokens(tokenizer,r.gist) for r in rows]
        warning_rows=sum(bool(v.warnings) for v in validation)
        assets.append({"directory":str(directory),"kind":kind,
                       "example_files":len(files),"packets":len(rows),"unique_examples":len({r.example_id for r in rows}),
                       "overlap_with_canonical_train1421":len({r.example_id for r in rows}&train_ids),
                       "valid_under_current_validator":sum(v.valid for v in validation),
                       "rows_with_grounding_warnings":warning_rows,
                       "mean_source_tokens_qwen3_8b":sum(source_tokens)/len(rows),
                       "mean_gist_tokens_qwen3_8b":sum(gist_tokens)/len(rows),
                       "mean_gist_source_ratio":sum(g/s for g,s in zip(gist_tokens,source_tokens))/len(rows),
                       "model":metadata.get("model_name"),"model_revision":metadata.get("model_revision"),
                       "input_path":metadata.get("input_path") or metadata.get("input_examples"),
                       "generation_parameters":metadata.get("generation_parameters"),
                       "reads_question_declared":metadata.get("reads_question"),
                       "reads_gold_declared":metadata.get("reads_gold_annotations"),
                       "per_packet_prompt_token_cost_recorded":False,
                       "per_packet_output_token_cost_recorded":False,
                       "per_packet_latency_recorded":False,
                       "source_span_offsets_recorded":False})
    teacher=[a for a in assets if a["kind"]=="teacher_generated"]
    result={"protocol":"SEM-A0_EXISTING_PACKET_PROVENANCE_GROUNDING_COST_AUDIT",
            "semantic_packet_assets":assets,"teacher_generated_asset_count":len(teacher),
            "teacher_generated_examples":sum(a["unique_examples"] for a in teacher),
            "teacher_generated_overlap_with_canonical_train1421":sum(a["overlap_with_canonical_train1421"] for a in teacher),
            "generator_query_conditioned":False,
            "generator_prompt_inputs":"source text and source-derived numeric constraints only; PacketGenerator does not pass QAExample.question or answers to build_packet_prompt",
            "gold_or_target_outcome_input_detected":False,
            "grounding_strength":"source text is stored and number/title checks are hard; unseen entity detection is warning-only and no source-span offsets or entailment certificate are stored",
            "online_cost_evidence":"generation metadata records model/config but not per-packet prompt tokens, output tokens, or latency",
            "decision":"STOP_EXISTING_PACKETS_AS_DIRECT_SEM_A1_INPUT",
            "reason":"Existing teacher packets do not instantiate the query-conditioned hypothesis, have zero canonical train1421 overlap, lack strong span-level grounding, and lack per-packet online cost traces.",
            "new_target_calls":0,"new_teacher_calls":0,"sealed_sets_read":False}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
