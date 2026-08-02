#!/usr/bin/env python3
import json, os, statistics, subprocess
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
for s in ("build_cpp_p1_2f.sh","build_cpp_p1_3c0.sh"): subprocess.run([str(root/"scripts"/s)],check=True)
subprocess.run([str(root/".venv/bin/python"),str(root/"scripts/export_qwen_lm_head.py")],check=True)
env=os.environ.copy(); c=root/"vendor/nvidia-535.288/usr/lib/x86_64-linux-gnu"; env["LD_LIBRARY_PATH"]=str(c)+(":"+env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
f=root/"artifacts/qwen-lm-head"; runs=[]; final_pairs=[]
for repeat in range(5):
  rec={}; initial=None; tail=f/f"tail-{repeat}.bin"
  for step in range(4):
    h=f/f"c3-h{step}.f32"; cm=f/f"c3-c{step}.json"; lm=f/f"c3-l{step}.json"
    cmd=[str(root/"build/cpp/solidattention-p1-2f"),"--fixture",str(root/"artifacts/qwen-28-layers"),"--plan",str(root/"artifacts/qwen-28-layers/chain-plan.tsv"),"--metrics",str(cm),"--hidden-output",str(h),"--position",str(511+step),"--resident-weights","--pipeline-kv","--final-audit-only","--read-ahead","16"]
    if initial: cmd += ["--initial-hidden",str(initial)]
    if step>0:
      cmd += ["--include-current-kv","--token-kv-output",str(tail),"--tail-tokens",str(step-1)]
      if step>1: cmd += ["--tail-kv-input",str(tail)]
    subprocess.run(cmd,check=True,env=env,stdout=subprocess.DEVNULL)
    subprocess.run([str(root/"build/cpp/solidattention-p1-3c0"),"--fixture",str(f),"--hidden",str(h),"--metrics",str(lm)],check=True,env=env)
    rec[f"step_{step}"]={"chain":json.loads(cm.read_text()),"lm":json.loads(lm.read_text()),"tail_bytes":tail.stat().st_size if step>0 else 0}
    if repeat==4: final_pairs.append((np.fromfile(h,np.float32),np.fromfile(f/"native_logits.f32",np.float32)))
    token=rec[f"step_{step}"]["lm"]["argmax_token"]; initial=f/f"embedding-{repeat}.f32"
    subprocess.run([str(root/"build/cpp/solidattention-p1-3c0"),"--fixture",str(f),"--hidden",str(h),"--metrics",str(f/"unused.json"),"--embedding-token",str(token),"--embedding-output",str(initial)],check=True,env=env,stdout=subprocess.DEVNULL)
  runs.append(rec)
norm=np.fromfile(f/"final_norm.f32",np.float32); w=np.memmap(f/"lm_head.f32",np.float32,"r",shape=(151936,1024)); audits=[]
for step,(h,native) in enumerate(final_pairs):
  x=h/np.sqrt(np.mean(h*h)+1e-6)*norm; teacher=np.asarray(w@x); d=np.abs(native-teacher)
  audits.append({"step":step,"native_token":int(native.argmax()),"teacher_token":int(teacher.argmax()),"max_abs_error":float(d.max()),"cosine":float(np.dot(native.astype(np.float64),teacher.astype(np.float64))/(np.linalg.norm(native.astype(np.float64))*np.linalg.norm(teacher.astype(np.float64))))})
source=(f/"tail-4.bin").read_bytes(); tail31=bytearray()
for layer in range(28):
  chunk=source[layer*3*4096:(layer+1)*3*4096]
  for token in range(31): tail31 += chunk[(token%3)*4096:(token%3+1)*4096]
tail31_path=f/"tail31-boundary.bin"; tail31_path.write_bytes(tail31); boundary_metrics=f/"tail32-boundary.json"; tail32=f/"tail32-boundary.bin"
subprocess.run([str(root/"build/cpp/solidattention-p1-2f"),"--fixture",str(root/"artifacts/qwen-28-layers"),"--plan",str(root/"artifacts/qwen-28-layers/chain-plan.tsv"),"--metrics",str(boundary_metrics),"--initial-hidden",str(f/"c3-h3.f32"),"--position","543","--resident-weights","--pipeline-kv","--final-audit-only","--read-ahead","16","--include-current-kv","--tail-kv-input",str(tail31_path),"--tail-tokens","31","--token-kv-output",str(tail32)],check=True,env=env,stdout=subprocess.DEVNULL)
boundary=json.loads(boundary_metrics.read_text())
summary={"version":"P1.3c.3-persistent-decode-tail","repeats":5,"steps":4,"tokens":[runs[-1][f"step_{s}"]["lm"]["argmax_token"] for s in range(4)],"tail_tokens":[runs[-1][f"step_{s}"]["chain"]["output_tail_tokens"] for s in range(4)],"tail_bytes":[runs[-1][f"step_{s}"]["tail_bytes"] for s in range(4)],"chain_wall_ms_median":[statistics.median(r[f"step_{s}"]["chain"]["total_wall_ms"] for r in runs) for s in range(4)],"kv_pack_mismatches":[runs[-1][f"step_{s}"]["chain"]["token_kv_pack_mismatches"] for s in range(4)],"logit_audits":audits,"raw_runs":runs}
summary["tail_31_to_32_boundary"]={"input_tokens":31,"output_tokens":boundary["output_tail_tokens"],"output_bytes":tail32.stat().st_size,"pack_mismatches":boundary["token_kv_pack_mismatches"],"wall_ms":boundary["total_wall_ms"]}
(root/"artifacts/runs/P1.3c.3-persistent-decode-tail-metrics.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps({k:v for k,v in summary.items() if k!="raw_runs"},indent=2))
