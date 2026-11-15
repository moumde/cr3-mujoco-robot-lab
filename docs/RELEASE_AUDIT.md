# P5b-2 release inventory and owner decisions

| Inventory | Candidate handling | Owner decision |
| --- | --- | --- |
| Original simulation code, generated configuration, documentation and CI | The local candidate includes the project [MIT License](../LICENSE). | Confirm the copyright attribution before publication. |
| External Python dependencies | Installed separately. The pinned environment metadata identify MuJoCo and LeRobot as Apache-2.0, SB3 as MIT, PyAV as BSD-3-Clause and Pillow as MIT-CMU. Other dependencies can contain multiple bundled licenses. | Review notices if binaries, wheels or a container will be distributed. |
| SB3-related training code | The PPO and SAC modules import and subclass SB3. `outputs/p5a-r2/research.py` implements the clipped surrogate and `outputs/p3c-10.7/conservative_sac.py` overrides the SAC update loop. No upstream package source tree was found in Git. | The [third-party notices](../THIRD_PARTY_NOTICES.md) record SB3 and reproduce its complete upstream MIT text. |
| Dobot CR3 files | Users fetch the exact upstream revision locally using the [asset script](../scripts/fetch_cr3.py). Verified SHA-256 values in the [asset manifest](../configs/assets.json). Top-level MIT and package BSD labeling are already recorded. | Confirm exact package and mesh terms before any redistribution of inputs, converted models or renders. |
| BC/PPO/SAC and other checkpoints | Local. Frozen hashes and protocols are tracked. | Confirm ownership and permission for each proposed weight bundle. |
| Raw trajectories, LeRobot videos, generated MP4/GIF and backup archives | Local. Public users with the fetched assets can generate new demonstrations on their own machine. The local release candidate tracks the generator and media index. | Confirm rights and credit requirements before sharing archived data, renders or trained artifacts. |
| Experiment statistics and figures | All seven R3 strategy records and numeric charts are suitable release material, subject to owner sign-off. The figures contain plots of recorded metrics. | Approve inclusion with project documentation. |
| Public hero video | The README links to the media index. The local 14.28 s cut is the chosen hero footage. | Supply an approved published URL or asset once the render can be distributed. |

The [R3 per-episode CSV](../outputs/p5a-r3/episodes.csv) is tracked, 252,721 bytes, contains 560 records and matches the archived SHA-256 `9ba45cc2a2c7ebd1aade9fa8b46061bdb008f44ab0d6392b69c9cb6a6d9f9170`. No new statistical substitute was needed. The release checker validates links and headings, tracked run scripts, frozen scene coverage, tracked asset hashes, archive-like file suffixes and obvious user-specific absolute paths. The audit of the candidate history found no weight, mesh, replay, video or backup archives under the audited file extensions. This is a local engineering inventory.

The CI workflow checks out the source, installs the pinned Public Core dependency set and fetches the pinned CR3 locally. It needs no private weight or dataset. Optional Research and Historical Full deliberately report missing input paths. See the [capability and test matrix](REPRODUCTION.md#presentation-and-test-tiers).

The [final freeze audit](P5B_3.md) lists the remaining owner decisions, including publication of Git author/committer metadata, together with the verified merge procedure.
