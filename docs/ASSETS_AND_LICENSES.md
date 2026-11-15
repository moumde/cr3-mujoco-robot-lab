# Assets and licenses

[Asset manifest](../configs/assets.json) gives the pinned source, version, checkout-relative path, SHA-256, local acquisition instructions and affected workflows. Use `python scripts/demo.py check quickstart` for the default demo or select `learned`, `research` or `vision-bc` for additional checks.

| Asset | Acquisition and distribution status |
| --- | --- |
| Dobot CR3 Xacro, package metadata and meshes | `python scripts/fetch_cr3.py` fetches upstream commit `7807ef8fb31e8f86a76cd474a4a9cddffef6eca9`. The upstream root is MIT and package metadata labels BSD with an unresolved variant. Treat inputs and generated model derivatives as local-only under the P0 review. |
| Scene and gripper source, calibration implementation | Original project source tracked in Git. Camera calibration is reconstructed from the pinned simulation and fixed camera configuration. |
| Frozen BC, Uniform Residual BC, PPO, SAC and vision BC | Produced by the project's simulated experiments. Restore from a private authorized copy with `demo.py restore`. SHA-256 pins the archived bytes. Public distribution review is pending. Re-running the historical training scripts is a separate reproduction exercise and may produce different bytes. |
| Raw data and LeRobot exports | Simulated demonstrations from project controllers using the CR3 asset. Use `demo.py collect` and `demo.py export` to build local examples. Historical visual data and private replay bundles require a private authorized copy. Rendered media also require distribution review. |

Dobot upstream: [official source at the pinned commit](https://github.com/Dobot-Arm/DOBOT_6Axis_ROS2_V4/tree/7807ef8fb31e8f86a76cd474a4a9cddffef6eca9).

The project's original source code uses the [MIT License](../LICENSE). The [third-party notices](../THIRD_PARTY_NOTICES.md) preserve the upstream Stable-Baselines3 license. Permissions for distributing external robot assets, archived renders, datasets and weights remain open. The local release candidate tracks source and lightweight numeric evidence. Private backups remain local.

See the [P5b-2 release inventory](RELEASE_AUDIT.md) for external Python provenance, media and specific owner decisions.
