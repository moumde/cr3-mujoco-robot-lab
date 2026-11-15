"""Plot PPO logs, compare frozen SAC and render a matched grasp episode."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch

from ppo import ROOT, ResidualPPO, BudgetedResidualEnv, GAIN, action, load_policy, model_at, panels
from conservative_sac import ConservativeSAC
from experiment import plant_for


def read(path):
    return json.loads(path.read_text())


def statistics(rows, baseline):
    done = [r for r in rows if r["is_success"]]
    result = {"success": len(done), "n": len(rows)}
    for metric in ("steps", "first_lift_step", "first_100mm_step"):
        values = [r[metric] for r in done]
        result[metric] = [float(np.mean(values)), *np.quantile(values,[.5,.95]).tolist()] if values else None
    delta = [r["steps"]-b["steps"] for r,b in zip(rows,baseline,strict=True)
             if r["is_success"] and b["is_success"]]
    result.update(delta_mean=float(np.mean(delta)) if delta else None,
                  faster=sum(d<0 for d in delta), slower=sum(d>0 for d in delta),
                  ik=sum(sum(r["ik_rejections"].values()) for r in rows),
                  contacts=sum(bool(r["collision"]) for r in rows),
                  absolute_mm=(1000*np.mean([r["absolute_m"] for r in rows],axis=0)).tolist(),
                  maximum_mm=(1000*np.max([r["absolute_m"] for r in rows],axis=0)).tolist())
    return result


def plot(out):
    history = ROOT.parent/"p3c-10.7-r1/run"
    base = read(history/"residual-bc.json")
    sac = read(history/"step-750.json")
    phases = {"BC": base, "SAC-750": sac}
    for path in sorted(out.glob("step-*.json")):
        phases["PPO-"+path.stem.split("-")[-1]] = read(path)
    metrics = {name:{part:statistics(rows[sl],base[sl]) for part,sl in
                     (("history",slice(0,42)),("validation",slice(42,62)))}
               for name,rows in phases.items()}
    online = [json.loads(l) for l in (out/"online.jsonl").read_text().splitlines()]
    updates = [json.loads(l) for l in (out/"updates.jsonl").read_text().splitlines()]
    sac_online = [json.loads(l) for l in (history/"online.jsonl").read_text().splitlines()]
    drift = {}
    for title, rows in (("PPO",online),("SAC",sac_online)):
        if rows:
            a=np.abs([r["policy_delta"] for r in rows])
            target=np.abs([r["target_delta_m"] for r in rows])*1000
            drift[title]=dict(raw_p95=np.quantile(a,.95,axis=0).tolist(),
                              projected_mm_p95=np.quantile(target,.95,axis=0).tolist(),
                              stochastic_saturation=(np.abs([r["stochastic"] for r in rows])>=.95).mean(0).tolist())
    if online:
        latest = read(out/"status.json")["checkpoints"][-1]["step"]
        final = ResidualPPO.load(out/f"model-{latest}.zip",device="cpu")
        # Rollout logs before the first update cannot measure post-update drift.
        differences = [final.predict(np.asarray(row["observation"],np.float32),
                                     deterministic=True)[0] - row["initialization"] for row in online]
        drift["PPO_final_on_logged_states"] = dict(
            raw_p95=np.quantile(np.abs(differences),.95,axis=0).tolist(),
            raw_max=np.max(np.abs(differences),axis=0).tolist())
    (out/"comparison.json").write_text(json.dumps(dict(metrics=metrics,drift=drift,
        online_endings=[(r["step"],r["scene_seed"],r["reason"]) for r in online
                        if r["terminated"] or r["truncated"]]),indent=2)+"\n")
    fig, axes = plt.subplots(2,4,figsize=(13,6))
    for ax,key in zip(axes.flat,("policy_gradient_loss","value_loss","approx_kl","clip_fraction",
                                "entropy_loss","log_std"),strict=False):
        ax.plot([r["step"] for r in updates],[r[key] for r in updates],marker="o")
        ax.set(title=key,xlabel="Online step")
    axes.flat[6].plot([r["step"] for r in online],np.cumsum([r["reward"] for r in online]))
    axes.flat[6].set(title="Cumulative online reward",xlabel="Online step")
    for part in ("history","validation"):
        ppo=[(int(k.split("-")[1]),v[part]["success"]/v[part]["n"])
             for k,v in metrics.items() if k.startswith("PPO")]
        axes.flat[7].plot(*zip(*ppo),marker="o",label=part)
    axes.flat[7].set(title="PPO success",xlabel="Online step",ylim=(0,1.03))
    axes.flat[7].legend()
    fig.tight_layout()
    fig.savefig(out/"training.png",dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(10,7))
    labels=list(metrics)
    axes[0,0].bar(labels,[r["history"]["steps"][0] if r["history"]["steps"] else 0 for r in metrics.values()])
    axes[0,0].set_title("Mean successful steps")
    axes[0,1].bar(labels,[r["history"]["ik"] for r in metrics.values()],label="IK")
    axes[0,1].bar(labels,[r["history"]["contacts"] for r in metrics.values()],label="Contact",alpha=.7)
    axes[0,1].legend()
    for label,rows in (("PPO",online),("SAC",sac_online)):
        axes[1,0].plot([r["step"] for r in rows],[np.linalg.norm(r["deterministic"]) for r in rows],label=label)
        axes[1,1].plot([r["step"] for r in rows],[np.linalg.norm(r["policy_delta"]) for r in rows],label=label)
    axes[1,0].set_title("Raw residual norm (online states)")
    axes[1,1].set_title("Same-state drift norm")
    for ax in axes.flat:
        if ax in axes[1]:
            ax.legend()
        else:
            ax.tick_params(axis="x",rotation=25)
    fig.tight_layout()
    fig.savefig(out/"comparison.png",dpi=140)


def media(out, seed=17208):
    torch.set_num_threads(1)
    case=next(c for c in panels() if c[1]==seed and c[2]=="flat_facing_camera")
    profile,seed,orientation,xy,_=case
    bc=load_policy()
    init=model_at(ROOT.parent/"p3c-10.5/uniform/model.pt")
    sac=ConservativeSAC.load(ROOT.parent/"p3c-10.7-r1/run/model-750.zip",device="cpu")
    latest=read(out/"status.json")["checkpoints"][-1]["step"]
    ppo=ResidualPPO.load(out/f"model-{latest}.zip",device="cpu")
    for label,predict in (("BC",lambda o:action(init,o,GAIN)),
                          ("SAC-750",lambda o:sac.predict(o,deterministic=True)[0]*GAIN),
                          (f"PPO-{latest}",lambda o:ppo.predict(o,deterministic=True)[0]*GAIN)):
        plant=plant_for(profile,xy)
        env=BudgetedResidualEnv(plant,bc)
        frames=[]
        obs,_=env.reset(seed=seed,options=dict(scene_seed=seed,orientation=orientation))
        with mujoco.Renderer(plant.model,height=320,width=480) as renderer:
            while True:
                obs,_,term,trunc,info=env.step(predict(obs))
                if info["steps"]%5==0 or term or trunc:
                    renderer.update_scene(plant.data,"overview")
                    img=Image.fromarray(renderer.render())
                    ImageDraw.Draw(img).text((8,8),f"{label} | {seed} flat | {info['steps']} | {info['reason'] or ''}",fill="white")
                    frames.append(img)
                if term or trunc:
                    break
        frames[0].save(out/f"{label}.gif",save_all=True,append_images=frames[1:],duration=100,loop=0)
        env.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--media",action="store_true")
    p.add_argument("--media-seed", type=int, choices=(17208, 17218), default=17208)
    a=p.parse_args()
    plot(a.output)
    if a.media:
        media(a.output, a.media_seed)
