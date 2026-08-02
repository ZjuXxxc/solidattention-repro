#!/usr/bin/env python3
import json, os, statistics, subprocess
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
for s in ("build_cpp_p1_2f.sh","build_cpp_p1_3c0.sh"): subprocess.run([str(root/"scripts"/s)],check=True)
subprocess.run([str(root/".venv/bin/python"),str(root/"scripts/export_qwen_lm_head.py")],check=True)
env=os.environ.copy(); c=root/"vendor/nvidia-535.288/usr/lib/x86_64-linux-gnu"; env["LD_LIBRARY_PATH"]=str(c)+(":"+env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
f=root/"artifacts/qwen-lm-head"; runs=[]; last=[]
for repeat in range(5):
  rec={}; initial=None
  for step in range(2):
    h=f/f"c2-h{step}.f32"; cm=f/f"c2-c{step}.json"; lm=f/f"c2-l{step}.json"
    cmd=[str(root/"build/cpp/solidattention-p1-2f"),"--fixture",str(root/"artifacts/qwen-28-layers"),"--plan",str(root/"artifacts/qwen-28-layers/chain-plan.tsv"),"--metrics",str(cm),"--hidden-output",str(h),"--position",str(511+step),"--resident-weights","--pipeline-kv","--final-audit-only","--read-ahead","16"]
    if initial: cmd += ["--initial-hidden",str(initial)]
    if step==1: cmd += ["--include-current-kv","--token-kv-output",str(f/"c2-current-token-kv.bin")]
    subprocess.run(cmd,check=True,env=env,stdout=subprocess.DEVNULL)
    subprocess.run([str(root/"build/cpp/solidattention-p1-3c0"),"--fixture",str(f),"--hidden",str(h),"--metrics",str(lm)],check=True,env=env)
    rec[f"step_{step}"]={"chain":json.loads(cm.read_text()),"lm":json.loads(lm.read_text())}
    if repeat==4: last.append((np.fromfile(h,np.float32),np.fromfile(f/"native_logits.f32",np.float32)))
    token=rec[f"step_{step}"]["lm"]["argmax_token"]; initial=f/"c2-embedding.f32"
    subprocess.run([str(root/"build/cpp/solidattention-p1-3c0"),"--fixture",str(f),"--hidden",str(h),"--metrics",str(f/"unused.json"),"--embedding-token",str(token),"--embedding-output",str(initial)],check=True,env=env,stdout=subprocess.DEVNULL)
  runs.append(rec)
norm=np.fromfile(f/"final_norm.f32",np.float32); w=np.memmap(f/"lm_head.f32",np.float32,"r",shape=(151936,1024)); audits=[]
for step,(h,native) in enumerate(last):
  x=h/np.sqrt(np.mean(h*h)+1e-6)*norm; teacher=np.asarray(w@x); d=np.abs(native-teacher)
  audits.append({"step":step,"native_token":int(native.argmax()),"teacher_token":int(teacher.argmax()),"max_abs_error":float(d.max()),"cosine":float(np.dot(native.astype(np.float64),teacher.astype(np.float64))/(np.linalg.norm(native.astype(np.float64))*np.linalg.norm(teacher.astype(np.float64))))})
summary={"version":"P1.3c.2-current-token-kv-attention","repeats":5,"tokens":[runs[-1][f"step_{s}"]["lm"]["argmax_token"] for s in range(2)],"step_chain_wall_ms_median":[statistics.median(r[f"step_{s}"]["chain"]["total_wall_ms"] for r in runs) for s in range(2)],"step_lm_wall_ms_median":[statistics.median(r[f"step_{s}"]["lm"]["native_wall_ms"] for r in runs) for s in range(2)],"step1_kv_bytes":(f/"c2-current-token-kv.bin").stat().st_size,"step1_kv_pack_mismatches":runs[-1]["step_1"]["chain"]["token_kv_pack_mismatches"],"logit_audits":audits,"raw_runs":runs}
(root/"artifacts/runs/P1.3c.2-current-token-kv-attention-metrics.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps({k:v for k,v in summary.items() if k!="raw_runs"},indent=2))
