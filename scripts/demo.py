"""Local CLI for simulation demos, data and frozen research artifacts."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from hashlib import file_digest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/assets.json"


def assets():
    return json.loads(MANIFEST.read_text())["assets"]


def checksum(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def verify(groups):
    errors = []
    for item in assets():
        if not set(item["required_for"]) & groups:
            continue
        path = ROOT / item["path"]
        if not path.is_file() or checksum(path) != item["sha256"]:
            errors.append(f"{item['path']} ({item['acquisition']})")
    if errors:
        raise SystemExit("Missing or mismatched assets:\n" + "\n".join(errors))


def restore(source, groups):
    # Validate every requested file before copying any bytes.
    plan = []
    for item in assets():
        if not set(item["required_for"]) & groups or item["acquisition"] == "Git checkout":
            continue
        src, dst = source / item["path"], ROOT / item["path"]
        if dst.is_file():
            if checksum(dst) != item["sha256"]:
                raise SystemExit(f"Refusing to overwrite mismatched asset: {dst}")
            continue
        if not src.is_file() or checksum(src) != item["sha256"]:
            raise SystemExit(f"Missing or mismatched source asset: {src}")
        plan.append((src,dst))
    for src,dst in plan:
        dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open("rb") as original, dst.open("xb") as copy:
            shutil.copyfileobj(original,copy)
    verify(groups)


def launch(python, *args):
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/cr3-mpl")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env["PYTHONPATH"] = str(ROOT/"src")
    subprocess.run([python,*map(str,args)],cwd=ROOT,env=env,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python",default=sys.executable)
    sub=p.add_subparsers(dest="command",required=True)
    for name in ("check","restore"):
        c=sub.add_parser(name)
        c.add_argument("groups",nargs="+",choices=("quickstart","data","learned","research","vision-bc"))
        if name=="restore":
            c.add_argument("--source",type=Path,required=True)
    for name in ("rgb","learned","collect","research","vision-bc","figures"):
        c=sub.add_parser(name)
        c.add_argument("output",type=Path)
        if name=="collect":
            c.add_argument("--episodes", type=int, default=2, help="Number of episodes (even, 2–40)")
            c.add_argument("--controller", choices=("rgb_expert", "rgb_learned_grasp"),
                           default="rgb_expert")
        if name=="research":
            c.add_argument("--panel",choices=("development","published-test"),default="development")
        if name=="vision-bc":
            c.add_argument("--manifest",type=Path,required=True,help="P4c original raw dataset root")
    for name in ("export","verify-data"):
        c=sub.add_parser(name)
        c.add_argument("raw",type=Path)
        c.add_argument("dataset",type=Path)
        c.add_argument("--lerobot-python",required=True)
    c=sub.add_parser("render-archive")
    c.add_argument("output",type=Path)
    c.add_argument("--archive",type=Path,required=True,help="Private P5a-R3 run directory")
    c=sub.add_parser("replay")
    c.add_argument("raw",type=Path)
    c.add_argument("--episode")
    c.add_argument("--gif", type=Path, help="Render the requested RGB Expert episode")
    a=p.parse_args()
    cmd=a.command
    if cmd in ("check","restore"):
        groups=set(a.groups)
        if groups & {"learned","research","vision-bc"}:
            groups.add("quickstart")
        (verify(groups) if cmd=="check" else restore(a.source.resolve(),groups))
        return
    if cmd in ("rgb","learned","research","vision-bc","collect","replay","render-archive"):
        verify({"quickstart",cmd})
    if cmd in ("rgb","learned"):
        launch(a.python,"-m","cr3_lab.tasks.p4a","--mode",
               "rgb_expert" if cmd=="rgb" else "rgb_learned_grasp",
               "--fixed-only","--output",a.output.resolve(),"--media")
    elif cmd=="collect":
        if not 2 <= a.episodes <= 40 or a.episodes % 2:
            p.error("--episodes must be an even number from 2 to 40")
        if a.controller == "rgb_learned_grasp":
            verify({"learned"})
        launch(a.python, "scripts/research_demo.py", "collect", a.output.resolve(),
               "--episodes", a.episodes, "--controller", a.controller)
    elif cmd in ("export","verify-data"):
        launch(a.lerobot_python,"-m","cr3_lab.demonstrations.export",
               "export" if cmd=="export" else "verify",a.raw.resolve(),a.dataset.resolve())
    elif cmd=="replay":
        if a.gif and not a.episode:
            p.error("--gif requires --episode")
        launch(a.python,"-m","cr3_lab.demonstrations.check",a.raw.resolve(),
               *(["--episode",a.episode] if a.episode else []))
        if a.gif:
            launch(a.python, "scripts/research_demo.py", "replay-gif", a.gif.resolve(),
                   "--raw", a.raw.resolve(), "--episode", a.episode)
    elif cmd=="render-archive":
        launch(a.python,"scripts/render_archive.py",a.archive.resolve(),a.output.resolve())
    elif cmd=="vision-bc":
        launch(a.python,"outputs/p4c/evaluate.py",a.manifest.resolve(),
               ROOT/"outputs/p4c/train/model.pt",a.output.resolve())
    else:
        launch(a.python,"scripts/research_demo.py",cmd,a.output.resolve(),
               *(["--panel",a.panel] if cmd=="research" else []))


if __name__=="__main__":
    main()
