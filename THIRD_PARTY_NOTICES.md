# Third-party notices

## Stable-Baselines3

Stable-Baselines3 2.7.1 is installed as a separate Python dependency and is distributed upstream under the MIT License. The project's residual PPO and SAC implementations extend its interfaces and include local training-loop implementations and overrides. This notice covers any material from Stable-Baselines3 used in those implementations, including the PPO research loop and conservative SAC training loop.

The original copyright and license text are reproduced in [Stable-Baselines3 license](LICENSES/Stable-Baselines3.txt). See the [upstream v2.7.1 source](https://github.com/DLR-RM/stable-baselines3/tree/v2.7.1), the [PPO research loop](outputs/p5a-r2/research.py) and the [conservative SAC loop](outputs/p3c-10.7/conservative_sac.py).

## Dobot CR3 inputs

The download helper retrieves robot model inputs directly from Dobot and stores them locally. The [asset inventory](docs/ASSETS_AND_LICENSES.md) records the pinned upstream revision and licensing notes for the model package. Distribution of archived renders, datasets and checkpoints is addressed separately in that inventory.
