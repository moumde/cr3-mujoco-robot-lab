"""Render complete matched research trajectories and plot TCP paths."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch

from research import (ROOT, GAIN, BudgetedResidualEnv, ResearchPPO, ResearchSAC,
                      action, load_policy, model_at, panels, plant_for)


def main():
    torch.set_num_threads(1)
    selections=json.loads((ROOT/"selected.json").read_text())
    case=next(c for c in panels() if c[1]==17218 and c[2]=="flat_facing_camera")
    profile,seed,orientation,xy,_=case
    bc=load_policy()
    reference=model_at(ROOT.parent/"p3c-10.5/uniform/model.pt")
    strategies=[("BC",lambda o:action(reference,o,GAIN))]
    for name,cls in (("ppo-11",ResearchPPO),("sac-11",ResearchSAC)):
        selected=selections[name]
        if not selected["step"]:
            continue
        policy=cls.load(ROOT/selected["path"],device="cpu")
        label=f"{name} {selected['step']} | {'eligible' if selected['deployable'] else 'research only'}"
        strategies.append((label,lambda o,model=policy:model.predict(o,deterministic=True)[0]*GAIN))
    out=ROOT/"media"
    out.mkdir()
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for index,(label,predict) in enumerate(strategies):
        env=BudgetedResidualEnv(plant_for(profile,xy),bc)
        plant=env.unwrapped
        obs,_=env.reset(seed=seed,options=dict(scene_seed=seed,orientation=orientation))
        frames=[]
        trace=[]
        try:
            with mujoco.Renderer(plant.model,height=320,width=480) as renderer:
                while True:
                    obs,_,term,trunc,info=env.step(predict(obs))
                    trace.append(info)
                    if info["steps"]%5==1 or term or trunc:
                        renderer.update_scene(plant.data,"overview")
                        img=Image.fromarray(renderer.render())
                        ImageDraw.Draw(img).text((8,8),f"{label} | 17218 flat | {info['steps']} | {info['reason'] or ''}",
                                                 fill="white",stroke_width=1,stroke_fill="black")
                        frames.append(img)
                    if term or trunc:
                        break
        finally:
            env.close()
        filename=f"strategy-{index}"
        frames[0].save(out/f"{filename}.gif",save_all=True,append_images=frames[1:],duration=100,loop=0)
        (out/f"{filename}.json").write_text(json.dumps(dict(label=label,trace=trace),indent=2)+"\n")
        tcp=1000*np.array([r["residual_budget"]["tcp_world_m"] for r in trace])
        rejected=np.array([r["ik_rejected"] for r in trace])
        axes[0].plot(tcp[:,0],tcp[:,2],label=label)
        axes[0].scatter(tcp[rejected,0],tcp[rejected,2],s=7)
        axes[1].plot(np.arange(len(tcp)),tcp[:,2],label=label)
    axes[0].set(xlabel="TCP X (mm)",ylabel="TCP Z (mm); points = IK rejected")
    axes[1].set(xlabel="Control step",ylabel="TCP Z (mm)")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out/"tcp.png",dpi=140)


if __name__=="__main__":
    main()
