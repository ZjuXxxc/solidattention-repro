#!/usr/bin/env python3
import json, os, statistics, subprocess
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
for script in ("build_cpp_p1_2f.sh","build_cpp_p1_3c0.sh"):
    subprocess.run([str(root/"scripts"/script)],check=True)
subprocess.run([str(root/".venv/bin/python"),str(root/"scripts/export_qwen_lm_head.py")],check=True)
env=os.environ.copy(); compat=root/"vendor/nvidia-535.288/usr/lib/x86_64-linux-gnu"
env["LD_LIBRARY_PATH"]=str(compat)+(":"+env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
f=root/"artifacts/qwen-lm-head"; runs=[]; last_logits=[]; last_hidden=[]
for repeat in range(5):
    record={}; initial=None
    for step in range(2):
        h=f/f"feedback-h{step}.f32"; cm=f/f"feedback-c{step}.json"; lm=f/f"feedback-l{step}.json"
        cmd=[str(root/"build/cpp/solidattention-p1-2f"),"--fixture",str(root/"artifacts/qwen-28-layers"),"--plan",str(root/"artifacts/qwen-28-layers/chain-plan.tsv"),"--metrics",str(cm),"--hidden-output",str(h),"--position",str(511+step),"--resident-weights","--pipeline-kv","--final-audit-only","--read-ahead","16"]
        if initial: cmd += ["--initial-hidden",str(initial)]
        subprocess.run(cmd,check=True,env=env,stdout=subprocess.DEVNULL)
        subprocess.run([str(root/"build/cpp/solidattention-p1-3c0"),"--fixture",str(f),"--hidden",str(h),"--metrics",str(lm)],check=True,env=env)
        record[f"step_{step}"]={"chain":json.loads(cm.read_text()),"lm":json.loads(lm.read_text())}
        if repeat==4:
            last_hidden.append(np.fromfile(h,np.float32)); last_logits.append(np.fromfile(f/"native_logits.f32",np.float32))
        token=record[f"step_{step}"]["lm"]["argmax_token"]
        initial=f/"feedback-embedding.f32"
        subprocess.run([str(root/"build/cpp/solidattention-p1-3c0"),"--fixture",str(f),"--hidden",str(h),"--metrics",str(f/"unused.json"),"--embedding-token",str(token),"--embedding-output",str(initial)],check=True,env=env,stdout=subprocess.DEVNULL)
    runs.append(record)
norm=np.fromfile(f/"final_norm.f32",np.float32); weight=np.memmap(f/"lm_head.f32",np.float32,"r",shape=(151936,1024))
audits=[]
for step,(hidden,native) in enumerate(zip(last_hidden,last_logits)):
    x=hidden/np.sqrt(np.mean(hidden*hidden)+1e-6)*norm; teacher=np.asarray(weight@x); d=np.abs(native-teacher)
    audits.append({"step":step,"native_token":int(native.argmax()),"teacher_token":int(teacher.argmax()),"max_abs_error":float(d.max()),"mean_abs_error":float(d.mean()),"cosine":float(np.dot(native.astype(np.float64),teacher.astype(np.float64))/(np.linalg.norm(native.astype(np.float64))*np.linalg.norm(teacher.astype(np.float64))))})
summary={"version":"P1.3c.1-two-token-native-feedback","repeats":5,"steps":2,
 "selection_policy":"fixed prompt block plan; decode KV not yet inserted",
 "tokens":[runs[-1][f"step_{s}"]["lm"]["argmax_token"] for s in range(2)],
 "step_chain_wall_ms_median":[statistics.median(r[f"step_{s}"]["chain"]["total_wall_ms"] for r in runs) for s in range(2)],
 "step_lm_wall_ms_median":[statistics.median(r[f"step_{s}"]["lm"]["native_wall_ms"] for r in runs) for s in range(2)],
 "logit_audits":audits,"raw_runs":runs}
path=root/"artifacts/runs/P1.3c.1-two-token-native-feedback-metrics.json"; path.write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps({k:v for k,v in summary.items() if k!="raw_runs"},indent=2))
