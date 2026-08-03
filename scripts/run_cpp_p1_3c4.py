#!/usr/bin/env python3
import json, os, subprocess, time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
for s in ("build_cpp_p1_2f.sh","build_cpp_p1_3c0.sh","build_cpp_p1_3c4.sh"): subprocess.run([str(root/"scripts"/s)],check=True)
subprocess.run([str(root/".venv/bin/python"),str(root/"scripts/export_qwen_lm_head.py")],check=True)
env=os.environ.copy();c=root/"vendor/nvidia-535.288/usr/lib/x86_64-linux-gnu";env["LD_LIBRARY_PATH"]=str(c)+(":"+env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
f=root/"artifacts/qwen-lm-head"; tail=f/"c4-tail.bin"; initial=None; tokens=[]; queries=[]; walls=[]; start=time.perf_counter()
for step in range(33):
  h=f/"c4-hidden.f32";cm=f/"c4-chain.json";lm=f/"c4-lm.json";qf=f/"c4-query.bin"
  cmd=[str(root/"build/cpp/solidattention-p1-2f"),"--fixture",str(root/"artifacts/qwen-28-layers"),"--plan",str(root/"artifacts/qwen-28-layers/chain-plan.tsv"),"--metrics",str(cm),"--hidden-output",str(h),"--position",str(511+step),"--resident-weights","--pipeline-kv","--final-audit-only","--read-ahead","16"]
  if initial:cmd += ["--initial-hidden",str(initial)]
  if step>0:
    cmd += ["--include-current-kv","--token-kv-output",str(tail),"--token-query-output",str(qf),"--tail-tokens",str(step-1)]
    if step>1:cmd += ["--tail-kv-input",str(tail)]
  subprocess.run(cmd,check=True,env=env,stdout=subprocess.DEVNULL); walls.append(json.loads(cm.read_text())["total_wall_ms"])
  subprocess.run([str(root/"build/cpp/solidattention-p1-3c0"),"--fixture",str(f),"--hidden",str(h),"--metrics",str(lm)],check=True,env=env)
  token=json.loads(lm.read_text())["argmax_token"];tokens.append(token);initial=f/"c4-embedding.f32"
  subprocess.run([str(root/"build/cpp/solidattention-p1-3c0"),"--fixture",str(f),"--hidden",str(h),"--metrics",str(f/"unused.json"),"--embedding-token",str(token),"--embedding-output",str(initial)],check=True,env=env,stdout=subprocess.DEVNULL)
  if step>0:queries.append(np.fromfile(qf,np.float32).reshape(28,16,128))
q=np.stack(queries,axis=2); qpath=f/"c4-queries-layer-head-token.f32";q.astype(np.float32).tofile(qpath)
seal=root/"artifacts/cpp-p1-3c4";subprocess.run([str(root/"build/cpp/solidattention-p1-3c4"),"--tail",str(tail),"--queries",str(qpath),"--output",str(seal)],check=True)
sm=json.loads((seal/"metrics.json").read_text());summary={"version":"P1.3c.4-online-seal-selection","steps":33,"generated_tail_tokens":32,"tokens":tokens,"tail_bytes":tail.stat().st_size,"chain_wall_ms_last":walls[-1],"end_to_end_wall_s":time.perf_counter()-start,"seal":sm,"representative_method":"InfLLM local-causal query-head mass, topk=4","selection_scope":"newly sealed block candidate; prompt representative merge pending"}
(root/"artifacts/runs/P1.3c.4-online-seal-selection-metrics.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps({**summary,"tokens":tokens[:4]+["..."]+tokens[-4:]},indent=2))
