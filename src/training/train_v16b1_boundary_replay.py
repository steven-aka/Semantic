from __future__ import annotations

import argparse,json,random,time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.schemas import read_jsonl,write_jsonl
from src.evaluation.sequential_policy_evaluation import evaluate_loaded_policy,head_state_dict,load_model
from src.model.sequential_policy_loss import sequential_policy_loss
from src.reproducibility import experiment_metadata,sha256,write_metadata
from src.training.sequential_policy_data import SequentialHistoryCollator,SequentialHistoryDataset


def collate_selected(items):
    rows=[x[0] for x in items];inputs=[x[1] for x in items];selected=[x[2] for x in items];flat=[(i,h) for i,values in enumerate(selected) for h in values];width=max(len(h["history"]) for _,h in flat)
    histories=torch.full((len(flat),width),-1,dtype=torch.long);lengths=torch.zeros(len(flat),dtype=torch.long);indices=torch.zeros(len(flat),dtype=torch.long);reached=torch.zeros(len(flat),dtype=torch.long);active=torch.zeros(len(flat),dtype=torch.long);optimal=[]
    for j,(i,h) in enumerate(flat):
        values=h["history"];histories[j,:len(values)]=torch.tensor(values);lengths[j]=len(values);indices[j]=i;reached[j]=h["reached_levels"];active[j]=len(rows[i]["active_levels"]);optimal.append(h["optimal_actions"])
    tokens=torch.tensor([r["packet_tokens"] for r in rows],dtype=torch.float32);fractions=tokens/tokens.sum(1,keepdim=True)
    return {"rows":rows,"inputs":inputs,"example_ids":[r["example_id"] for r in rows],"histories":histories,"history_lengths":lengths,"example_indices":indices,"reached_levels":reached,"active_level_counts":active,"optimal_actions":optimal,"packet_token_fractions":fractions}


def sample_batch(dataset,boundary_indices,retention_indices,rng,queries_per_half=4,histories_per_query=44):
    output=[];targets=[]
    for index in rng.sample(boundary_indices,queries_per_half):
        row,inputs=dataset[index];values=row["boundary_histories"];chosen=[rng.choice(values) for _ in range(histories_per_query)];targets.append(chosen);output.append((row,inputs,chosen))
    for index,target in zip(rng.sample(retention_indices,queries_per_half),targets):
        row,inputs=dataset[index];values=row["retention_histories"];chosen=[]
        for wanted in target:
            distance=[abs(len(x["history"])-len(wanted["history"]))+abs(int(x["reached_levels"])-int(wanted["reached_levels"])) for x in values];best=min(distance);chosen.append(rng.choice([x for x,d in zip(values,distance) if d==best]))
        output.append((row,inputs,chosen))
    rng.shuffle(output);return collate_selected(output)


def diagnostic(model,dataset,device,pad,seed):
    rng=random.Random(seed);groups={"boundary":[],"retention":[]};model.eval()
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        for kind,key in (("boundary","boundary_histories"),("retention","retention_histories")):
            indices=[i for i,r in enumerate(dataset.rows) if r[key]]
            for start in range(0,len(indices),8):
                items=[]
                for i in indices[start:start+8]:
                    row,inputs=dataset[i];values=row[key];items.append((row,inputs,[rng.choice(values) for _ in range(min(44,len(values)))]))
                b=collate_selected(items);logits,progress,_,_=model.forward_states(b["inputs"],b["histories"].to(device),b["history_lengths"].to(device),b["example_indices"].to(device),b["packet_token_fractions"].to(device),pad_token_id=pad,device=device);result=sequential_policy_loss(logits,progress,b["optimal_actions"],b["reached_levels"].to(device),b["active_level_counts"].to(device),progress_weight=.2);groups[kind].append((len(b["optimal_actions"]),float(result.action),result.action_accuracy))
    return {k:{"states":sum(x[0] for x in v),"action_loss":sum(x[0]*x[1] for x in v)/sum(x[0] for x in v),"top1_set_accuracy":sum(x[0]*x[2] for x in v)/sum(x[0] for x in v)} for k,v in groups.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument("--protocol-config",required=True);p.add_argument("--train-data",required=True);p.add_argument("--internal-validation-data",required=True);p.add_argument("--development-data",required=True);p.add_argument("--exact-dir",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--model",default="models/Qwen3-4B");p.add_argument("--model-dim",type=int,default=512);p.add_argument("--max-sequence-length",type=int,default=512);p.add_argument("--max-optimizer-steps",type=int,default=250);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--learning-rate",type=float,default=2e-4);p.add_argument("--weight-decay",type=float,default=.01);p.add_argument("--warmup-steps",type=int,default=25);p.add_argument("--progress-weight",type=float,default=.2);p.add_argument("--seed",type=int,default=20260912);a=p.parse_args()
    protocol=json.loads(Path(a.protocol_config).read_text());expected=protocol["arguments"]
    if protocol["status"]!="APPROVED_TO_RUN":raise ValueError("protocol not approved")
    mismatch={k:(getattr(a,k),v) for k,v in expected.items() if getattr(a,k)!=v}
    if mismatch:raise ValueError(mismatch)
    random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);device=torch.device("cuda:0");out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);write_metadata(out/"protocol_snapshot.json",protocol);started=time.perf_counter()
    tokenizer,model=load_model(a)
    for parameter in model.lm.parameters():parameter.requires_grad=False
    params=list(model.head_parameters());train=SequentialHistoryDataset(list(read_jsonl(a.train_data)),tokenizer,a.max_sequence_length);valid=SequentialHistoryDataset(list(read_jsonl(a.internal_validation_data)),tokenizer,a.max_sequence_length);boundary=[i for i,r in enumerate(train.rows) if r["boundary_histories"]];retention=[i for i,r in enumerate(train.rows) if r["retention_histories"]]
    optimizer=torch.optim.AdamW(params,lr=a.learning_rate,weight_decay=a.weight_decay)
    from transformers import get_cosine_schedule_with_warmup
    scheduler=get_cosine_schedule_with_warmup(optimizer,a.warmup_steps,a.max_optimizer_steps);rng=random.Random(a.seed);history=[]
    for step in range(1,a.max_optimizer_steps+1):
        model.train();b=sample_batch(train,boundary,retention,rng)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logits,progress,_,_=model.forward_states(b["inputs"],b["histories"].to(device),b["history_lengths"].to(device),b["example_indices"].to(device),b["packet_token_fractions"].to(device),pad_token_id=int(tokenizer.pad_token_id),device=device);result=sequential_policy_loss(logits,progress,b["optimal_actions"],b["reached_levels"].to(device),b["active_level_counts"].to(device),progress_weight=a.progress_weight)
        result.total.backward();torch.nn.utils.clip_grad_norm_(params,1);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
        if step%25==0:record={"optimizer_step":step,"total_loss":float(result.total),"action_loss":float(result.action),"progress_loss":float(result.progress),"set_valued_action_accuracy":result.action_accuracy};history.append(record);write_jsonl(out/"training_history.jsonl",history);print(json.dumps(record),flush=True)
    torch.save(head_state_dict(model),out/"sequential_head.pt");diag=diagnostic(model,valid,device,int(tokenizer.pad_token_id),a.seed);write_metadata(out/"internal_validation_diagnostic.json",diag)
    development=SequentialHistoryDataset(list(read_jsonl(a.development_data)),tokenizer,a.max_sequence_length);loader=DataLoader(development,batch_size=a.batch_size,shuffle=False,collate_fn=SequentialHistoryCollator(a.seed));summary,details=evaluate_loaded_policy(model,loader,exact_dir=a.exact_dir,pad_token_id=int(tokenizer.pad_token_id),device=device,beam_width=8,progress_weight=a.progress_weight);write_jsonl(out/"development_details.jsonl",details);write_metadata(out/"development_summary.json",summary);write_metadata(out/"metadata.json",experiment_metadata(stage="v16b1_boundary_focused_replay",base_checkpoint=a.checkpoint,train_sha256=sha256(a.train_data),optimizer_steps=a.max_optimizer_steps,backbone_and_lora_frozen=True,complete_sequential_head_trainable=True,elapsed_seconds=time.perf_counter()-started,locked_roles_used=False,development300_used_once=True));print(json.dumps({"internal_validation":diag,"development":summary},indent=2))


if __name__=="__main__":main()
