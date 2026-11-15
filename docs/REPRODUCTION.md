# Reproduction

## Level 1 — simulation and demos

Follow the [Quick Start](../README.md#quick-start) and install `requirements.txt` for the demo interpreter. Use `requirements-core.txt` for Public Core CI and its test suite. The default run has no policy weights. Every output should use a new directory. Commands accept absolute paths and resolve source and default assets from the checkout.

To access the hybrid demo, restore from an authorized local source tree containing the original checkout-relative files.

```bash
python scripts/demo.py restore learned --source /path/to/private-asset-tree
python scripts/demo.py learned outputs/demo-learned
```

Collect two fixed-orientation episodes, or request 4–40 episodes in even increments to include random XY pairs. The first random pair is held out for validation. Add `--controller rgb_learned_grasp` to record the hybrid policy using the same interface.

```bash
python scripts/demo.py collect outputs/demo-raw --episodes 6
python scripts/demo.py replay outputs/demo-raw
python scripts/demo.py replay outputs/demo-raw --episode episode-000 --gif outputs/replayed.gif
```

The LeRobot environment used for validation has `lerobot==0.7.0` and MuJoCo 4.0.1. Keep LeRobot's PyTorch dependencies isolated from the demo interpreter. In a separate Python 3.12 environment install `lerobot==0.7.0` and `mujoco==4.0.1`; point `--lerobot-python` to that environment's interpreter. Set `HF_HOME` to a writable directory if the default cache is read-only. Video encoding and decoding need its PyAV/FFmpeg backend. The `verify-data` command checks the resulting data through the actual dataset loader.

```bash
python scripts/demo.py export outputs/demo-raw outputs/demo-lerobot --lerobot-python /path/to/lerobot-env/bin/python
python scripts/demo.py verify-data outputs/demo-raw outputs/demo-lerobot --lerobot-python /path/to/lerobot-env/bin/python
```

Raw data contain RGB, control streams and private Oracle diagnostics. The LeRobot default features contain only the two RGB inputs, robot state and executed actions. One exported action starts at the image timestamp. Use the raw log to reconstruct action chunks or exact dynamics replay.

## Level 2 — frozen research

```bash
python scripts/demo.py figures outputs/rebuilt-figures
python scripts/demo.py restore research --source /path/to/private-asset-tree
python scripts/demo.py research outputs/research-development
python scripts/demo.py render-archive outputs/rebuilt-gifs --archive /private/p5a-r3/run
```

The graphs use tracked P5a-R2 learning summaries and P5a-R3 evaluation metrics. Archive rendering requires the original P5a-R3 per-strategy `.jsonl.gz` traces, `.json` results and saved snapshots. The default research panel uses a two-scenario subset of the original development panel and evaluates all seven frozen strategies. The already-revealed 80-scenario panel can be reproduced with `--panel published-test`. Such reruns are replications on a known set and cannot support a new independent-test claim. Model identities and selection remain those of the published manifest. [Final protocol](P5A_R3.md)

The P4c checkpoint is a failure-analysis example. Supply the original raw dataset directory containing its manifest and episode folders. A newly collected dataset can exercise the API, while the original 0/8 result requires the original evaluation scenarios.

```bash
python scripts/demo.py restore vision-bc --source /path/to/private-asset-tree
python scripts/demo.py vision-bc outputs/vision-bc-replay --manifest /private/p4c/raw
```

## Verification layers

- Run `python scripts/demo.py check quickstart` and `PYTHONPATH=src MUJOCO_GL=egl python -m pytest -q tests/test_release_demo.py tests/test_p0.py tests/test_p4a.py`. Some P4a checks need local BC assets.
- Run the data export/readback and raw replay commands above to verify physical time, camera pairs and encoding.
- Run asset verification, development evaluation and archive rendering for the research layer. The legacy phase scripts under `outputs/` are tracked. Their large inputs and artifacts remain local, so older historical experiments may require additional manifests and backups.

## Presentation and test tiers

| Workflow | Tracked source alone | Additional local inputs |
| --- | --- | --- |
| Inspect numeric evidence, articles and plots | Yes | None |
| Public Core and RGB Expert Quick Start | Fetch pinned CR3 input files with the included script | Headless OpenGL for rendered demos |
| Collect fresh demonstrations and replay at 50 Hz | Same as Quick Start | LeRobot 0.7.0 interpreter with PyAV to export/read v3 |
| Optional Research and hybrid grasp | Uses the same simulator | Authorized Frozen BC, Residual BC and selected PPO/SAC checkpoints |
| Vision BC and historical full tests | Uses tracked source and protocols | Original vision dataset or historical logs and Teacher dataset as applicable |
| Rebuild the review MP4s | Composition script is tracked | Authorized P5b-1 validation root with source GIFs, control logs and LeRobot export |

```bash
python scripts/check_release.py
python scripts/test_release.py public-core
python scripts/test_release.py optional-research --output outputs/research-verification
python scripts/test_release.py historical-full
MPLCONFIGDIR=/tmp/cr3-mpl python scripts/presentation.py charts outputs/presentation-charts --archive /path/to/authorized/historical/outputs
/path/to/lerobot-env/bin/python scripts/presentation.py videos outputs/presentation-videos --archive /path/to/p5b-1-validation
```

Optional Research includes a fixed development-panel rollout. Historical Full additionally requires the P3c-7 `results.json`, P3c-10.4 `data/` and P3c-10.7 `run/online.jsonl`. Asset checks fail with the paths to restore. Video generation uses the local PyAV/libx264 encoder and archived media without advancing the simulator. The tracked charts can also be inspected directly via the [README](../README.md#rl-experiments).
