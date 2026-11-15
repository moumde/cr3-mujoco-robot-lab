"""Freeze the reserved panel, then execute it exactly once into a fresh directory."""
import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "p5a-r2"))
from research import (GAIN, BudgetedResidualEnv, ResearchPPO, ResearchSAC,
                      action, digest, load_policy, model_at, phase, plant_for)
from benchmark import TEST, sample_xy
from cr3_lab.tasks.motor_scene import Orientation

LIMIT = np.array([20., 5., 22.])
R2 = ROOT.parent / "p5a-r2"


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    with path.open("x") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write("\n")


def freeze():
    selected = read(R2 / "selected.json")
    expected = {"ppo-11": 2000, "ppo-22": 5000, "ppo-33": 20000,
                "sac-11": 2000, "sac-22": 5000, "sac-33": 5000}
    assert {k: v["step"] for k, v in selected.items()} == expected
    candidates = []
    for name, row in selected.items():
        algo, seed = name.split("-")
        path = R2 / row["path"]
        assert digest(path) == row["sha256"] and row["deployable"]
        review = next(r for r in read(R2/name/"checks.json") if r["step"] == row["step"])
        assert not review["review"]
        candidates.append(dict(name=name, algorithm=algo, seed=int(seed), step=row["step"],
            path=str(path.relative_to(Path.cwd())) if path.is_relative_to(Path.cwd()) else
            "outputs/p5a-r2/"+row["path"], sha256=row["sha256"], qualified=True,
            zero_reference=f"outputs/p5a-r2/{name}/step-0.json"))
    reference = read(ROOT.parent/"p3c-10.6/selected.json")
    assert reference["normalized_residual_gain"] == GAIN
    assert digest(Path(reference["checkpoint"])) == reference["checkpoint_sha256"]
    candidates.insert(0, dict(name="bc", algorithm="bc", seed=None, step=0,
        path=reference["checkpoint"], sha256=reference["checkpoint_sha256"], qualified=True,
        zero_reference="outputs/p3c-10.7-r1/run/residual-bc.json"))
    base = Path("outputs/p3c-6/run/model.pt")
    assert digest(base) == "999e77405fb4bfdc1dc43c1ef6b8a4cef48e4cf6924ffdd9c5d918b2dc668617"
    write(ROOT/"frozen_candidates.json", dict(candidates=candidates, initialization=reference,
        base=dict(path=str(base),sha256=digest(base),normalization="SCALE and checkpoint center/spread; clip ±8"),
        residual_normalization="P3c-10.5 checkpoint center/spread; copied to both actors",
        protocol="outputs/p5a-r2/protocol.json", protocol_sha256=digest(R2/"protocol.json"),
        qualification=read(R2/"protocol.json")["selection"], review_mm=LIMIT.tolist()))
    source = ROOT.parent/"p3c-7/results.json"
    seeds = read(source)["reserved_test_seeds"]["35"]
    assert seeds == list(TEST[35]) == list(range(17700,17740))
    write(ROOT/"test_manifest.json", dict(source="outputs/p3c-7/results.json",
        source_sha256=digest(source), width_m=.035, control_hz=50, max_steps=400,
        cases=[dict(seed=s,orientation=str(o),xy=sample_xy(.035,s).tolist())
               for s in seeds for o in Orientation]))


def integration(env):
    plant = env.unwrapped
    flag = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(plant.model, flag))
    mujoco.mj_getState(plant.model, plant.data, state, flag)
    return state


@torch.no_grad()
def execute(out):
    torch.set_num_threads(1)
    frozen = read(ROOT/"frozen_candidates.json")
    assert digest(R2/"protocol.json") == frozen["protocol_sha256"]
    assert digest(Path(frozen["base"]["path"])) == frozen["base"]["sha256"]
    manifest = read(ROOT/"test_manifest.json")
    assert digest(Path(manifest["source"])) == manifest["source_sha256"]
    for row in frozen["candidates"]:
        assert digest(Path(row["path"])) == row["sha256"]
    out.mkdir(parents=True)
    write(out/"started.json",dict(freeze=digest(ROOT/"frozen_candidates.json"),
                                  manifest=digest(ROOT/"test_manifest.json")))
    bc = load_policy()
    ref = model_at(Path(frozen["initialization"]["checkpoint"]))
    starts = {}
    for candidate in frozen["candidates"]:
        name = candidate["name"]
        cls = {"ppo": ResearchPPO, "sac": ResearchSAC}.get(candidate["algorithm"])
        model = cls.load(candidate["path"],device="cpu") if cls else None
        result = []
        for case in manifest["cases"]:
            seed, orientation, xy = (case[k] for k in ("seed","orientation","xy"))
            env = BudgetedResidualEnv(plant_for("35mm",xy),bc)
            scene = (seed,orientation)
            try:
                obs,_ = env.reset(seed=seed,options=dict(scene_seed=seed,orientation=orientation))
                state=integration(env)
                fingerprint=sha256(state.tobytes()+obs.tobytes()).hexdigest()
                assert starts.setdefault(scene,fingerprint)==fingerprint
                states=[state]
                log=[]
                counts=Counter()
                stages=Counter()
                total=0.
                drift=[]
                tcp=0.
                while True:
                    reference=action(ref,obs,1.)
                    raw=model.predict(obs,deterministic=True)[0] if model else reference
                    assert np.isfinite(raw).all() and np.max(np.abs(raw)) <= 1
                    stage=phase(obs)
                    after,reward,term,trunc,info=env.step(raw*GAIN)
                    trace=info["residual_budget"]
                    assert np.isfinite(after).all() and np.isfinite(reward)
                    assert info["reason"]!="simulation_fault"
                    assert np.all(np.abs(trace["signed_budget_m"])<=env.budget_m+1e-12)
                    delta=np.abs(raw-reference)
                    drift.append(delta)
                    counts[trace["ik_status"]]+=1
                    stages[stage]+=1
                    total+=reward
                    tcp+=np.linalg.norm(trace["measured_tcp_delta_m"])
                    log.append(dict(obs=obs.tolist(),next_obs=after.tolist(),policy=raw.tolist(),
                        reference=reference.tolist(),phase=stage,reward=reward,terminated=term,
                        truncated=trunc,**info))
                    # Save exact integration states for later offline scene rendering.
                    if info["steps"]%5==0 or term or trunc:
                        states.append(integration(env))
                    obs=after
                    if term or trunc:
                        break
                prefix=f"{name}-{seed}-{orientation}"
                with gzip.open(out/f"{prefix}.jsonl.gz","xt",compresslevel=2) as stream:
                    for row in log:
                        stream.write(json.dumps(row,allow_nan=False)+"\n")
                np.savez_compressed(out/f"{prefix}.npz", states=np.asarray(states))
                absolute=1000*np.asarray(trace["absolute_budget_m"])
                result.append(dict(candidate=name,seed=seed,orientation=orientation,xy=xy,
                    initial_sha256=fingerprint,steps=info["steps"],success=info["is_success"],
                    reason=info["reason"],final_stage=stage,collision=info["collision"],
                    bilateral=info["first_bilateral_step"],grasp=info["first_lift_step"],
                    lift=info["first_100mm_step"],hold=info["hold_steps"],reward=total,
                    ik=dict(counts),phase_steps=dict(stages),tcp_mm=1000*tcp,
                    drift_p95=np.quantile(drift,.95,axis=0).tolist(),
                    absolute_mm=absolute.tolist(),review_exceeded=bool(np.any(absolute>LIMIT))))
            finally:
                env.close()
        write(out/f"{name}.json",result)
        print(name,sum(r["success"] for r in result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("freeze","run"))
    parser.add_argument("--out",type=Path,default=ROOT/"run")
    args=parser.parse_args()
    freeze() if args.mode=="freeze" else execute(args.out)
