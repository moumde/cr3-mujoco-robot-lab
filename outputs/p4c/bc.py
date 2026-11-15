"""Small RGB + proprioception action-chunk BC, with one fixed training budget."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Sequential(nn.Conv2d(6, 16, 5, 2, 2), nn.ReLU(),
                                    nn.Conv2d(16, 24, 3, 2, 1), nn.ReLU(),
                                    nn.Conv2d(24, 32, 3, 2, 1), nn.ReLU(), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(788, 128), nn.ReLU(), nn.Linear(128, 128),
                                  nn.ReLU(), nn.Linear(128, 35))
        for key, size, fill in (("sx", 20, 0), ("ss", 20, 1), ("ax", 35, 0), ("as_", 35, 1)):
            self.register_buffer(key, torch.full((size,), float(fill)))

    def forward(self, image, state):
        return self.head(torch.cat((self.vision(image.float() / 255),
                                    (state - self.sx) / self.ss), dim=1))


def load(root, split):
    with np.load(root / f"{split}-cache.npz") as values:
        return tuple(torch.from_numpy(values[k]) for k in ("images", "state", "action", "phase"))


def train(root, out, device):
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(11)
    started = perf_counter()
    train_data, validation = (load(root, s) for s in ("train", "validation"))
    model = Policy()
    state, target = train_data[1:3]
    model.sx.copy_(state.mean(0))
    model.ss.copy_(state.std(0).clamp_min(.001))
    model.ax.copy_(target.mean(0))
    model.as_.copy_(target.std(0).clamp_min(.015))
    model.to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    best = float("inf")
    history = []

    @torch.no_grad()
    def measure(group):
        images, states, actions, phases = group
        errors, losses = [], []
        for start in range(0, len(images), 128):
            sl = slice(start, start+128)
            pred = model(images[sl].to(device), states[sl].to(device))
            error = pred - (actions[sl].to(device) - model.ax) / model.as_
            losses.append(error.square().cpu())
            errors.append((error * model.as_).abs().reshape(-1, 5, 7).cpu())
        errors = torch.cat(errors)
        return dict(mse=torch.cat(losses).mean().item(), mae=errors.mean((0, 1)).tolist(),
                    phase_mae={i: errors[phases == i].mean((0, 1)).tolist()
                               for i in torch.unique(phases).tolist()})

    for epoch in range(1, 31):
        for indices in torch.randperm(len(state)).split(128):
            images, states, actions = (a[indices].to(device) for a in train_data[:3])
            pred = model(images, states)
            loss = (pred - (actions - model.ax) / model.as_).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        row = dict(epoch=epoch, train=measure(train_data), validation=measure(validation))
        history.append(row)
        if row["validation"]["mse"] < best:
            best = row["validation"]["mse"]
            torch.save(model.state_dict(), out / "model.pt")
        print(epoch, row["train"]["mse"], row["validation"]["mse"], flush=True)
    (out / "training.json").write_text(json.dumps(dict(history=history, seed=11, device=device,
        epochs=30, learning_rate=.001, batch=128, best=min(history, key=lambda r:r["validation"]["mse"]),
        train_frames=len(state), validation_frames=len(validation[0]),
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
        parameters=sum(p.numel() for p in model.parameters()), wall_s=perf_counter()-started), indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = p.parse_args()
    train(a.dataset, a.output, a.device)
