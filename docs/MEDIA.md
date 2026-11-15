# Demonstration media

The review MP4s contain simulated CR3 renders and are kept in the local review bundle pending permission to distribute. The filenames below identify the clips. The publishing location will be set after approval.

| Clip | Local filename | Source and selection |
| --- | --- | --- |
| Complete visual manipulation | `a-complete.mp4` | P5b-1 fixed dual orientations, both RGB Expert and RGB + Residual BC. Shows the actual RGB estimate and all motion phases. |
| README cut | `a-quick.mp4` | Same two RGB Expert runs with frame decimation for 3× playback. |
| Learned grasp comparison | `b-learned-comparison.mp4` | Frozen R3 scene 17700, protrusion towards Inspection, BC / PPO 11 @2k / SAC 11 @2k all succeed. |
| Demonstration and replay | `c-dataset.mp4` | Same archived episode across physical replay, both raw RGB streams, sampled 50 Hz control curves and decoded LeRobot video. |
| SAC failure | `d-sac-failure-17721.mp4` | Frozen R3 scene 17721, flat face towards Inspection. Residual BC and PPO 11 succeed, SAC 33 @5k times out. TCP dots indicate rejected IK steps. |

The [frozen media selection](../outputs/p5a-r3/media-selection.json) and [all 560 episodes](../outputs/p5a-r3/episodes.csv) are tracked. The comparison uses the approved P5a-R3 archived trajectories. The rendered comparison holds each final frame after its episode ends. Generate permitted local footage and rebuild the review pack with the commands in the [reproduction guide](REPRODUCTION.md#presentation-and-test-tiers).

The local review bundle also includes a revised dataset clip with the overview visible at 0.00 s. The original P5b-2 clip remains archived. See the [replay validation](P5B_3_MEDIA.md).
