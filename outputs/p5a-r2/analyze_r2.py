"""Aggregate the registered experiment, select models and generate figures."""
from collections import Counter
import gzip
from hashlib import file_digest
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text())


def choose(records):
    eligible=[r for r in records if r["eligible"] and r["step"]]
    if eligible:
        return min(eligible,key=lambda r:(-r["validation"]["success"],r["validation"]["successful_mean"],r["step"]))
    return min([r for r in records if r["step"]] or records,
               key=lambda r:(-r["validation"]["success"],r["step"]))


def metrics(rows,base,returns):
    success=[r for r in rows if r["is_success"]]
    paired=[r["steps"]-p["steps"] for r,p in zip(rows,base,strict=True) if r["is_success"] and p["is_success"]]
    return dict(success=len(success),count=len(rows),all_mean=float(np.mean([r["steps"] for r in rows])),
        successful_mean=float(np.mean([r["steps"] for r in success])) if success else None,
        paired_n=len(paired),paired_delta=float(np.mean(paired)) if paired else None,
        reward_mean=float(np.mean(returns)),bilateral=sum(r["first_bilateral_step"] is not None for r in rows),
        confirmed=sum(r["first_lift_step"] is not None for r in rows),
        lift100=sum(r["first_100mm_step"] is not None for r in rows),
        ik=sum(sum(r["ik_rejections"].values()) for r in rows),
        collisions=sum(bool(r["collision"]) for r in rows),
        absolute_max_mm=(1000*np.max([r["absolute_m"] for r in rows],axis=0)).tolist(),
        failures=dict(Counter((r["reason"]+" / "+("hold" if r["first_100mm_step"] else "lift"
            if r["first_lift_step"] else "grasp" if r["first_bilateral_step"] else "approach"))
            for r in rows if not r["is_success"])))


def online_metrics(path, checkpoints):
    """Stream step logs into checkpoint bins without keeping full plant traces in RAM."""
    segments={}
    if not path.exists():
        return segments
    index=1
    data=[]
    with gzip.open(path,"rt") as stream:
        for line in stream:
            row=json.loads(line)
            while index<len(checkpoints) and row["step"]>checkpoints[index]:
                segments[checkpoints[index]]=segment(data)
                index+=1
                data=[]
            data.append(row)
    if data and index<len(checkpoints):
        segments[checkpoints[index]]=segment(data)
    return segments


def segment(rows):
    return dict(steps=len(rows),reward=float(np.mean([r["reward"] for r in rows])),
        actor_drift_p95=np.quantile(np.abs([r["policy_delta"] for r in rows]),.95,axis=0).tolist(),
        sample_abs_p95=np.quantile(np.abs([r["stochastic"] for r in rows]),.95,axis=0).tolist(),
        projected_mm_p95=(1000*np.quantile(np.abs([r["target_delta_m"] for r in rows]),.95,axis=0)).tolist(),
        actual_tcp_mm_p95=(1000*np.quantile(np.abs(
            [r["residual_budget"]["measured_tcp_delta_m"] for r in rows]),.95,axis=0)).tolist())


def analyze():
    protocol=read(ROOT/"protocol.json")
    baseline=read(ROOT.parent/"p3c-10.7-r1/run/residual-bc.json")
    summary,selected={},{}
    for algorithm in protocol["algorithms"]:
        for seed in protocol["seeds"]:
            name=f"{algorithm}-{seed}"
            folder=ROOT/name
            status=read(folder/"status.json")
            records=[]
            for chk in status["checks"]:
                step=chk["step"]
                rows=read(folder/f"step-{step}.json")
                returns=[]
                total=0.
                with gzip.open(folder/f"step-{step}.jsonl.gz","rt") as stream:
                    for line in stream:
                        tick=json.loads(line)
                        total+=tick["reward"]
                        if tick["terminated"] or tick["truncated"]:
                            returns.append(total)
                            total=0.
                assert len(returns)==62
                records.append(dict(**chk,eligible=not chk["review"],
                    history=metrics(rows[:42],baseline[:42],returns[:42]),
                    validation=metrics(rows[42:],baseline[42:],returns[42:])))
            summary[name]=dict(status=status,records=records,
                online=online_metrics(folder/"online.jsonl.gz",protocol["checkpoints"]))
            if records:
                chosen=choose(records)
                path=folder/f"model-{chosen['step']}.zip"
                with path.open("rb") as stream:
                    sha=file_digest(stream,"sha256").hexdigest()
                selected[name]=dict(step=chosen["step"],deployable=bool(chosen["eligible"] and chosen["step"]),
                    sha256=sha,path=str(path.relative_to(ROOT)),last_step=records[-1]["step"])
    spread={}
    for algorithm in protocol["algorithms"]:
        spread[algorithm]={}
        for step in protocol["checkpoints"]:
            rows=[r for seed in protocol["seeds"] for r in summary[f"{algorithm}-{seed}"]["records"]
                  if r["step"]==step]
            if rows:
                spread[algorithm][step]={part:{
                    "mean":float(np.mean([r[part]["success"] for r in rows])),
                    "min":min(r[part]["success"] for r in rows),
                    "max":max(r[part]["success"] for r in rows),
                } for part in ("validation","history")}
    (ROOT/"seed-spread.json").write_text(json.dumps(spread,indent=2)+"\n")
    (ROOT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (ROOT/"selected.json").write_text(json.dumps(selected,indent=2)+"\n")
    fig, axes=plt.subplots(3,3,figsize=(14,10))
    specs=[("validation","success"),("history","success"),("validation","reward_mean"),
           ("validation","paired_delta"),("history","ik"),("history","collisions"),
           ("validation","absolute_max_mm"),("online","actor_drift_p95"),("online","reward")]
    for ax,(split,metric) in zip(axes.flat,specs,strict=True):
        for algorithm,color in (("ppo","C0"),("sac","C1")):
            for seed in protocol["seeds"]:
                record=summary[f"{algorithm}-{seed}"]
                series=([(int(k),v[metric]) for k,v in record["online"].items()] if split=="online"
                        else [(r["step"],r[split][metric]) for r in record["records"]])
                if series:
                    x,y=zip(*series)
                    y=[max(v) if isinstance(v,list) else np.nan if v is None else v for v in y]
                    ax.plot(x,y,marker="o",color=color,alpha=.45+.2*(seed//11-1),label=f"{algorithm}-{seed}")
        ax.set(title=f"{split} {metric}",xlabel="online steps")
    for i,part in enumerate(("validation","history")):
        ax=axes.flat[i]
        for algorithm,color in (("ppo","C0"),("sac","C1")):
            items=list(spread[algorithm].items())
            if items:
                ax.fill_between([step for step,_ in items],
                                [r[part]["min"] for _,r in items],
                                [r[part]["max"] for _,r in items],alpha=.12,color=color)
    axes[0,0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT/"comparison.png",dpi=140)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for algorithm,panels in (("ppo",("approx_kl","clip_fraction","value")),
                             ("sac",("actor","critic","entropy"))):
        for seed in protocol["seeds"]:
            path=ROOT/f"{algorithm}-{seed}/updates.jsonl"
            if not path.exists():
                continue
            rows=[json.loads(l) for l in path.read_text().splitlines()]
            for j,key in enumerate(panels):
                data=[(r["step"],r[key]) for r in rows if key in r]
                if data:
                    ax=axes[0 if algorithm=="ppo" else 1,j]
                    ax.plot(*zip(*data),label=str(seed),alpha=.75)
                    ax.set(title=f"{algorithm} {key}",xlabel="online steps")
    for ax in axes.flat:
        if ax.lines:
            ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT/"optimization.png",dpi=140)
    plt.close(fig)
    breakdown={}
    for group,entry in summary.items():
        totals=Counter()
        for row in entry["records"]:
            totals.update(row["validation"]["failures"])
        breakdown[group]=dict(totals)
    tags=sorted({tag for failures in breakdown.values() for tag in failures})
    fig,ax=plt.subplots(figsize=(10,4))
    bottom=np.zeros(len(breakdown))
    for tag in tags:
        values=[d.get(tag,0) for d in breakdown.values()]
        ax.bar(list(breakdown),values,bottom=bottom,label=tag)
        bottom+=values
    if tags:
        ax.legend(fontsize=7)
    ax.set(ylabel="Validation failures across checkpoints")
    fig.tight_layout()
    fig.savefig(ROOT/"failures.png",dpi=140)


if __name__=="__main__":
    analyze()
