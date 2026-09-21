"""Evaluate the QAMPARI-adapted abstractive RECOMP checkpoint on Screen-64."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import src.baselines.frontier_r1_recomp_abstractive as base

def main():
    p=argparse.ArgumentParser();p.add_argument("--compressor-model",default="checkpoints/baselines/recomp_qampari_abstractive");p.add_argument("--output",type=Path,default=Path("results/v2_rank_then_cut/baseline_recomp_qampari_abstractive/screen64"));p.add_argument("--device",default="cuda");p.add_argument("--batch-size",type=int,default=2);p.add_argument("--target-batch-size",type=int,default=64);p.add_argument("--gpu-memory-utilization",type=float,default=.40);p.add_argument("--mode",choices=("compress","evaluate","all"),default="all")
    a=p.parse_args();base.OUT=a.output;rows=base.rows()
    if a.mode in ("compress","all"):base.compress(a,rows)
    if a.mode in ("evaluate","all"):
        base.evaluate(a,rows)
        path=a.output/"summary.json";summary=json.loads(path.read_text())
        summary.update({"protocol":"BASELINE-RECOMP-QAMPARI-ABSTRACTIVE-V1-SCREEN64","qualification":"RECOMP-style QAMPARI in-domain teacher distillation; not an official QAMPARI checkpoint","compressor_model":a.compressor_model,"decision":"RETAIN_NEGATIVE_FROZEN_TARGET_BASELINE"})
        path.write_text(json.dumps(summary,indent=2)+"\n")
if __name__=="__main__":main()
