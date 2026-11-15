"""Read-only P4c preprocessing parity and on/off-expert action audit."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

P4C = Path(__file__).resolve().parents[1] / "p4c"
sys.path.insert(0, str(P4C))
from bc import Policy
from evaluate import predict


def audit():
    torch.set_num_threads(1)
    model = Policy()
    model.load_state_dict(torch.load(P4C/"train/model.pt", map_location="cpu", weights_only=True))
    model.eval()
    with np.load(P4C/"dataset/validation-cache.npz") as cache, torch.no_grad():
        chunks = []
        for image, state in zip(cache["images"], cache["state"], strict=True):
            # Undo CHW packing; resizing already cached 48x32 does not change pixels.
            rgb = image.reshape(2, 3, 32, 48).transpose(0, 2, 3, 1)
            actual = predict(model, rgb, state).reshape(35)
            expected = model(torch.from_numpy(image)[None], torch.from_numpy(state)[None])
            np.testing.assert_array_equal(actual, (expected*model.as_+model.ax).numpy()[0])
            chunks.append(actual)
        array = np.asarray(chunks).reshape(-1,5,7)
        errors = np.abs(array-cache["action"].reshape(-1,5,7))
        raw = (np.asarray(chunks)-model.ax.numpy())/model.as_.numpy()
        result = dict(validation_frames=len(array), mae=errors.mean((0,1)).tolist(),
                      raw_range=[float(raw.min()),float(raw.max())],
                      closure_range=[float(array[:,:,6].min()),float(array[:,:,6].max())],
                      closure_oob=int(((array[:,:,6]<0)|(array[:,:,6]>1)).sum()),
                      J6_phase_mae={int(p):float(errors[cache["phase"]==p,:,5].mean())
                                    for p in np.unique(cache["phase"])})
    results = {}
    for identifier in ("episode-000","episode-001"):
        folder=P4C/"raw"/identifier
        controls=[json.loads(l) for l in (folder/"control.jsonl").read_text().splitlines()][:30]
        student=[json.loads(l) for l in (P4C/"eval"/f"{identifier}.jsonl").read_text().splitlines()][:30]
        with np.load(folder/"frames.npz") as frames:
            expert = np.concatenate([predict(model,img,np.asarray(controls[i*5]["state"],np.float32))
                                    for i,img in enumerate(frames["rgb"][:6])])
        results[identifier] = dict(
            expert_logged=[r["action"] for r in controls],
            predicted_on_expert=expert.tolist(), predicted_on_student=[r["requested"] for r in student])
    return dict(validation=result, failed_first_30=results)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    a.output.write_text(json.dumps(audit(),indent=2)+"\n")
