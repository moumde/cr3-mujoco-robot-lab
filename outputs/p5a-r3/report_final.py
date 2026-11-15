"""Summarize independent results and render recorded integration states offline."""
import csv
import gzip
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from final_test import ROOT, R2, read, write, plant_for

RUN = ROOT / "run"


def summarize(rows, base):
    success = [r["steps"] for r in rows if r["success"]]
    pairs = [r["steps"]-b["steps"] for r,b in zip(rows,base,strict=True) if r["success"] and b["success"]]
    return dict(n=len(rows),success=len(success),
        bilateral=sum(r["bilateral"] is not None for r in rows),
        grasp=sum(r["grasp"] is not None for r in rows),lift=sum(r["lift"] is not None for r in rows),
        hold=sum(r["hold"]>=20 for r in rows),signed_budget_violations=0,timeouts=sum(r["reason"]=="timeout" for r in rows),
        contacts=sum(bool(r["collision"]) for r in rows),review=sum(r["review_exceeded"] for r in rows),
        lost_baseline=sum(b["success"] and not r["success"] for r,b in zip(rows,base,strict=True)),
        eligible=not any(r["review_exceeded"] or r["collision"] or (b["success"] and not r["success"])
                         for r,b in zip(rows,base,strict=True)),
        successful_mean=float(np.mean(success)) if success else None,
        successful_p50_p95=np.quantile(success,[.5,.95]).tolist() if success else None,
        all_mean=float(np.mean([r["steps"] for r in rows])),paired_n=len(pairs),
        paired_mean=float(np.mean(pairs)) if pairs else None,
        faster=sum(d<0 for d in pairs),slower=sum(d>0 for d in pairs),
        ik={k:sum(r["ik"].get(k,0) for r in rows)
            for k in ("solver_nonconverged","joint_step_limit")},
        absolute_mean_mm=np.mean([r["absolute_mm"] for r in rows],axis=0).tolist(),
        absolute_max_mm=np.max([r["absolute_mm"] for r in rows],axis=0).tolist(),
        drift_episode_p95_mean=np.mean([r["drift_p95"] for r in rows],axis=0).tolist(),
        phase_steps=dict(sum((Counter(r["phase_steps"]) for r in rows),Counter())),
        reward_mean=float(np.mean([r.get("reward",0.) for r in rows])),
        tcp_mean_mm=float(np.mean([r["tcp_mm"] for r in rows])),
        failures=dict(Counter(f"{r['reason']} / {r['final_stage']}" for r in rows if not r["success"])))


def graphics(results, all_rows):
    names=list(results)
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for ax,key in zip(axes.flat,("success","paired_mean","ik","absolute_max_mm","failures","review"),strict=True):
        if key in ("ik","failures"):
            keys=sorted(set().union(*(r[key] for r in results.values())))
            bottom=np.zeros(len(names))
            for label in keys:
                heights=[r[key].get(label,0) for r in results.values()]
                ax.bar(names,heights,bottom=bottom,label=label)
                bottom+=heights
            if keys:
                ax.legend(fontsize=6)
        elif key=="absolute_max_mm":
            for i,axis in enumerate("XYZ"):
                ax.bar(np.arange(len(names))+(i-1)*.24,[r[key][i] for r in results.values()],.24,label=axis)
            ax.set_xticks(range(len(names)),names)
            ax.legend()
        else:
            ax.bar(names,[r[key] or 0 for r in results.values()])
        ax.set_title(key)
        ax.tick_params(axis="x",rotation=30)
    fig.tight_layout()
    fig.savefig(ROOT/"independent.png",dpi=150)
    plt.close(fig)
    source=read(R2/"summary.json")
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for name,run in source.items():
        algo=name.split("-")[0]
        for ax,field in ((axes[0,0],"history"),(axes[0,1],"validation")):
            ax.plot([r["step"] for r in run["records"]],[r[field]["success"]/r[field]["count"] for r in run["records"]],label=name)
            ax.set(title=f"R2 {field}",xlabel="Online steps")
        with (R2/name/"updates.jsonl").open() as stream:
            import json
            rows=[json.loads(l) for l in stream]
        ax=axes[1,0 if algo=="ppo" else 1]
        for key in (("approx_kl","clip_fraction") if algo=="ppo" else ("actor","critic")):
            data=[(r["step"],r[key]) for r in rows if key in r]
            ax.plot(*zip(*data),label=f"{name} {key}",alpha=.7)
        if algo=="sac":
            data=[(int(step),max(r["actor_drift_p95"])) for step,r in run["online"].items()]
            ax.plot(*zip(*data),linestyle="--",label=f"{name} drift")
        ax.set(title=f"R2 {algo} optimization",xlabel="Online steps")
    for ax in axes.flat:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(ROOT/"historical-learning.png",dpi=150)


def render(label,index,group,rows,metadata):
    case=rows[group[0]][index]
    seed,orientation=case["seed"],case["orientation"]
    fig,ax=plt.subplots(figsize=(6,4))
    videos=[]
    plant=plant_for("35mm",case["xy"])
    flag=mujoco.mjtState.mjSTATE_INTEGRATION
    with mujoco.Renderer(plant.model,height=240,width=360) as renderer:
        for name in group:
            row=rows[name][index]
            trace=RUN/f"{name}-{seed}-{orientation}"
            with gzip.open(str(trace)+".jsonl.gz","rt") as stream:
                import json
                steps=[json.loads(l) for l in stream]
            tcp=1000*np.asarray([s["residual_budget"]["tcp_world_m"] for s in steps])
            ax.plot(tcp[:,0],tcp[:,2],label=name)
            reject=np.array([s["ik_rejected"] for s in steps])
            ax.scatter(tcp[reject,0],tcp[reject,2],s=4)
            frames=[]
            with np.load(str(trace)+".npz") as saved:
                for state in saved["states"]:
                    mujoco.mj_setState(plant.model,plant.data,state,flag)
                    mujoco.mj_forward(plant.model,plant.data)
                    renderer.update_scene(plant.data,"overview")
                    image=Image.fromarray(renderer.render())
                    title=f"{name} @{metadata[name]['step']} | {seed} {orientation.split('_')[0]}"
                    verdict=f"{row['reason']} | review {'over' if row['review_exceeded'] else 'within'}"
                    draw=ImageDraw.Draw(image)
                    draw.rectangle((0,0,360,34),fill="black")
                    draw.text((6,3),title,fill="white")
                    draw.text((6,18),verdict,fill="white")
                    frames.append(image)
            videos.append(frames)
    combined=[]
    for i in range(max(map(len,videos))):
        img=Image.new("RGB",(360*len(group),240))
        for j,frames in enumerate(videos):
            img.paste(frames[min(i,len(frames)-1)],(360*j,0))
        combined.append(img)
    combined[0].save(ROOT/f"{label}.gif",save_all=True,append_images=combined[1:],duration=100,loop=0)
    ax.set(xlabel="TCP X (mm)",ylabel="TCP Z (mm), dots are IK rejects")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT/f"{label}-tcp.png",dpi=150)
    plt.close(fig)
    return dict(seed=seed,orientation=orientation,strategies=group)


def main():
    metadata={r["name"]:r for r in read(ROOT/"frozen_candidates.json")["candidates"]}
    rows={name:read(RUN/f"{name}.json") for name in metadata}
    for group in rows.values():
        assert len(group)==80
        for a,b in zip(group,rows["bc"],strict=True):
            assert (a["seed"],a["orientation"],a["initial_sha256"])==(b["seed"],b["orientation"],b["initial_sha256"])
    results={name:summarize(group,rows["bc"]) for name,group in rows.items()}
    aggregate={}
    for algo in ("ppo","sac"):
        members=[v for k,v in results.items() if k.startswith(algo)]
        aggregate[algo]={key:[float(np.mean([v[key] for v in members])),
                              min(v[key] for v in members),max(v[key] for v in members)]
                         for key in ("success","successful_mean","paired_mean","review")}
    with (ROOT/"episodes.csv").open("x") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows["bc"][0]),lineterminator="\n")
        writer.writeheader()
        for group in rows.values():
            writer.writerows(group)
    by_orientation={}
    for name,group in rows.items():
        by_orientation[name]={}
        for orientation in sorted({r["orientation"] for r in group}):
            idx=[i for i,r in enumerate(group) if r["orientation"]==orientation]
            by_orientation[name][orientation]=summarize([group[i] for i in idx],
                                                        [rows["bc"][i] for i in idx])
    write(ROOT/"summary.json",dict(strategies=results,seed_mean_min_max=aggregate,
                                  by_orientation=by_orientation))
    write(ROOT/"complete.json",dict(pairs=80,strategies=7,steps=sum(r["steps"] for g in rows.values() for r in g)))
    graphics(results,rows)
    group=["bc","ppo-11","sac-11"]
    ok=next(i for i in range(80) if all(rows[n][i]["success"] for n in group))
    clips={"shared-success":render("shared-success",ok,group,rows,metadata)}
    failures=[(i,n) for n in rows if n!="bc" for i,b in enumerate(rows["bc"]) if b["success"] and not rows[n][i]["success"]]
    if failures:
        i,name=max(failures,key=lambda v:sum(rows[v[1]][v[0]]["ik"].get(k,0)
                                           for k in ("solver_nonconverged","joint_step_limit")))
        rivals=["bc",name,next(n for n in rows if n.startswith("sac-" if name.startswith("ppo") else "ppo-") and rows[n][i]["success"])] if any(
            n.startswith("sac-" if name.startswith("ppo") else "ppo-") and rows[n][i]["success"] for n in rows) else ["bc",name]
        clips["regression"]=render("regression",i,rivals,rows,metadata)
    eligible=[i for i in range(80) if i != ok and all(rows[n][i]["success"] for n in group)]
    if eligible:
        contrast=max(eligible,key=lambda i: np.ptp([rows[n][i]["tcp_mm"] for n in group]))
        clips["tcp-contrast"]=render("tcp-contrast",contrast,group,rows,metadata)
    write(ROOT/"media-selection.json",clips)


if __name__=="__main__":
    main()
