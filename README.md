# Procedural Content Generation for 2D Platformers

## proc_gen_ppo
 
This repository contains the source code for the project - Procedural Content Generation for 2D Platformers.

- A Reinforcement Learning (RL) model based on Proximal Policy Optimization (PPO) algorithm that can create levels for 2D platformers.
- A custom environment (with rendering capabilities) for the reinforcement learning model to interact during the generation of levels. The environment wrapped with the Gym for compatibility.

## Functionalities
- The RL algorithm generates grid-based levels and exports them in a json format.
- The exported levels (json) can then be loaded in any game engine or other software for evaluation or play testing.

## Evaluation
- The RL model was evaluated on <b>100 randomly generated player paths</b> with constraints.
- The evaluation also involved 2 other random-based approaches for performance comparison.
- The RL model achieved <b>52% solvability rate</b>, significantly outperforming random generation methods.
- The exported levels were further evaluated by a Unity-based platformer game for human feedback.
- All levels that were marked solvable by the algorithm were fully solvable and playable in the client game.


!["path_complexities_and_solvabilities"](./saves/visualizations/Recurrent%20PPO%20(Proposed%20approach)/path_complexities.png)

Following image shows a sample of the level generated.

### Legend
- Squares - Cells representing input player path (This path represents the approximate route the player in the actual game should follow).
- Pink circles - All cells that are reachable by the player character.
- Yellow circles - Hanging cells - Cells that are valid, by no path leading to it, thereby non-reachable.

!["sample level"](./saves/evaluation/proposed_approach/test_path_77_img.png)
