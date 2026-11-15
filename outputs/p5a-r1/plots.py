"""Plot paired 17218 paths and empirical PPO updates without retraining."""
import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ppo_update_audit import ROOT, HISTORY, ResidualPPO, load_policy, model_at, panels, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    from ppo import torch
    torch.set_num_threads(1)
    out=args.output
    out.mkdir()
    checkpoint=json.loads((ROOT/"run/status.json").read_text())["checkpoints"][-1]["step"]
    old,new,latest=[ResidualPPO.load(path,device="cpu") for path in
        (HISTORY/"model-0.zip",HISTORY/"model-250.zip",ROOT/"run"/f"model-{checkpoint}.zip")]
    bc,ref=load_policy(),model_at(ROOT.parent/"p3c-10.5/uniform/model.pt")
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for i,case in enumerate(c for c in panels() if c[1]==17218):
        trajectories=trace(out,case,bc,ref,old,new,latest)
        for name in ("bc","ppo250","r1"):
            rows=trajectories[name]
            tcp=1000*np.array([r["residual_budget"]["tcp_world_m"] for r in rows])
            rejected=np.array([r["ik_rejected"] for r in rows])
            axes[i,0].plot(tcp[:,0],tcp[:,2],label=name)
            if rejected.any():
                axes[i,0].scatter(tcp[rejected,0],tcp[rejected,2],s=8)
            absolute=np.array([r["residual_budget"]["absolute_budget_m"] for r in rows])*1000
            axes[i,1].plot(absolute,label=[name+" "+axis for axis in "XYZ"])
        axes[i,0].set(title=str(case[2]),xlabel="TCP X (mm)",ylabel="TCP Z (mm); dots = IK rejected")
        axes[i,1].set(xlabel="Control step",ylabel="Absolute command travel (mm)")
        for ax in axes[i]:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out/"paths.png",dpi=140)
    fig, axes=plt.subplots(2,2,figsize=(9,6))
    for folder,name in ((HISTORY,"P5a"),(ROOT/"run","R1")):
        rows=[json.loads(l) for l in (folder/"updates.jsonl").read_text().splitlines()]
        for ax,key in zip(axes.flat,("approx_kl","clip_fraction","policy_gradient_loss","value_loss"),strict=True):
            ax.plot([r["step"] for r in rows],[r[key] for r in rows],marker="o",label=name)
            ax.set(title=key,xlabel="Online step")
            ax.legend()
    fig.tight_layout()
    fig.savefig(out/"updates.png",dpi=140)


if __name__=="__main__":
    main()
