"""Run small frozen comparisons or render tracked statistics without weights."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text())


def collect(out, count, controller):
    from cr3_lab.demonstrations.collect import collect as record
    from cr3_lab.tasks.motor_scene import SpawnRange

    configs = [(SpawnRange.FIXED, 5400, False, controller, "train")]
    configs += [(SpawnRange.LARGE, seed, False, controller,
                 "validation" if seed == 5401 else "train")
                for seed in range(5401, 5400 + count//2)]
    record(out, ROOT/"outputs", configs=configs)


def research(out, panel):
    import torch
    sys.path.insert(0, str(ROOT/"outputs/p5a-r3"))
    from final_test import execute, GAIN, ResearchPPO, ResearchSAC, action, load_policy, model_at
    from research import rollout

    torch.set_num_threads(1)
    if panel == "published-test":
        execute(out)
        return
    frozen = read(Path("outputs/p5a-r3/frozen_candidates.json"))
    # Use tracked copies of the measured original ±35 mm developer scenarios.
    dev = read(Path("configs/development-demo.json"))
    out.mkdir(parents=True)
    bc = load_policy()
    base = model_at(Path(frozen["initialization"]["checkpoint"]))
    rows = []
    import gzip
    with gzip.open(out/"trace.jsonl.gz","xt") as stream:
        for item in frozen["candidates"]:
            cls = {"ppo":ResearchPPO, "sac":ResearchSAC}.get(item["algorithm"])
            model = cls.load(item["path"],device="cpu") if cls else None
            def policy(obs):
                return model.predict(obs,deterministic=True)[0]*GAIN if model else action(base,obs,GAIN)
            for case in dev:
                _, result = rollout(case,bc,item["name"],policy,stream)
                rows.append(dict(strategy=item["name"],checkpoint=item["step"],**result))
    (out/"results.json").write_text(json.dumps(rows,indent=2)+"\n")


def figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True)
    source=read(Path("outputs/p5a-r2/summary.json"))
    test=read(Path("outputs/p5a-r3/summary.json"))["strategies"]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,run in source.items():
        for ax,panel in zip(axes,("history","validation"),strict=True):
            rs=run["records"]
            ax.plot([r["step"] for r in rs],[r[panel]["success"]/r[panel]["count"] for r in rs],
                    marker="o",label=name)
            ax.set(title=panel,xlabel="Online environment steps",ylabel="Stable success rate")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out/"learning.png",dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,key,label in zip(axes.flat,("success","paired_mean","ik","absolute_max_mm"),
            ("Successes / 80","Paired delta steps vs BC","IK rejections","Maximum absolute XYZ (mm)"),
            strict=True):
        if key=="absolute_max_mm":
            for i,axis in enumerate("XYZ"):
                ax.plot(list(test),[r[key][i] for r in test.values()],marker="o",label=axis)
            ax.legend()
        else:
            values=[sum(r[key].values()) if key=="ik" else r[key] for r in test.values()]
            ax.bar(list(test),values)
        ax.set_title(label)
        ax.tick_params(axis="x",rotation=30)
    fig.tight_layout()
    fig.savefig(out/"independent.png",dpi=150)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("collect","research","figures","replay-gif"))
    p.add_argument("output",type=Path)
    p.add_argument("--episodes",type=int,default=2)
    p.add_argument("--controller", default="rgb_expert")
    p.add_argument("--panel",default="development")
    p.add_argument("--raw", type=Path)
    p.add_argument("--episode")
    a=p.parse_args()
    if a.mode=="collect":
        collect(a.output, a.episodes, a.controller)
    elif a.mode=="research":
        research(a.output,a.panel)
    elif a.mode == "replay-gif":
        sys.path.insert(0, str(ROOT / "outputs/p4c"))
        from visualize_run import replay

        if a.output.exists():
            raise FileExistsError(a.output)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        replay(a.raw, a.episode, a.output)
    else:
        figures(a.output)
