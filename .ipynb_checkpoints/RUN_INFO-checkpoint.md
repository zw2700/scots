# Run Info

## Hierarchical Planner Command

From the repository root, run the hierarchical planner directly with:

```bash
MUJOCO_GL=egl python pipelines/scots/scots_ogbench.py task=pointmaze-medium-stitch-v0
```

For evaluation:

```bash
MUJOCO_GL=egl python pipelines/scots/scots_ogbench.py task=pointmaze-medium-stitch-v0 mode=inference
```

Useful overrides:

```bash
MUJOCO_GL=egl python pipelines/scots/scots_ogbench.py task=antmaze-medium-stitch-v0 dataset_source=none
MUJOCO_GL=egl python pipelines/scots/scots_ogbench.py task=antmaze-medium-stitch-v0 dataset_source=only
MUJOCO_GL=egl python pipelines/scots/scots_ogbench.py task=antmaze-medium-stitch-v0 dataset_source=concat
```

`mode=train` is the default. `dataset_source` supports `none`, `only`, and `concat`, and the default is `none`.

## What `dataset_source=none` Means

If you run `scots_ogbench.py` with `dataset_source=none`, the script does not load SCoTS-augmented datasets from the stitcher outputs. It trains the hierarchical planner only on the original OGBench offline dataset.

This means you do not need to train the inverse dynamics model, HILP representation, or stitcher before training the hierarchical planner itself when `dataset_source=none`.

However, evaluation still requires a pretrained low-level controller checkpoint, because `mode=inference` loads a low-level controller based on the task config. The current code supports `gciql`, `crl`, and `gcbc`.

## High-Level Training and Evaluation Flow

The description below matches how the hierarchical planner works for supported antmaze tasks such as `antmaze-medium-stitch-v0`. A repo task config has also been added for `antmaze-medium-navigate-v0`.

### Training with `dataset_source=none`

1. The script loads the original OGBench dataset for the selected task.
2. Because `dataset_source=none`, it skips loading any augmented `.npz` files from `results/stitcher_ogbench_H...`.
3. The dataset wrapper keeps only the XY part of the observation, normalizes it, and builds multi-horizon trajectory slices.
4. The code trains two diffusion models:
   - a high-level planner over coarse waypoint sequences
   - a low-level planner over shorter XY segments between adjacent waypoints

For `antmaze-medium-stitch-v0`, the task config uses:

```yaml
env_name: "antmaze-medium-stitch-v0"
horizons: [21, 26]
low_controller: crl
goal_tol: 3.0
replan_every: 50
low_horizon: 5
w_ldg: 0
```

At a high level, the hierarchical diffusion planner learns to model feasible XY waypoint plans from the offline dataset only.

### Evaluation with `dataset_source=none`

1. The script loads the trained hierarchical diffusion checkpoints.
2. It also loads a pretrained low-level controller checkpoint:
   - `results/GCIQL/<env>/...` for `gciql`
   - `results/GCBC/<env>/...` for `gcbc`
   - `results/CRL/<env>/...` for `crl`
3. At the start of each episode, the environment provides the current state and the task goal.
4. The planner uses only the XY coordinates of the current state and final goal to sample:
   - a high-level sequence of waypoints
   - low-level XY segments connecting consecutive waypoints
5. Those segments are stitched into one long XY plan.
6. The low-level controller converts the current full observation plus the current XY subgoal into an action.
7. The system advances to the next subgoal when it is within `goal_tol`, and replans every `replan_every` environment steps.

The important dependency split is:

- Training with `dataset_source=none` does not require SCoTS augmentation artifacts.
- Evaluation still requires the low-level controller checkpoint.

## Current Recommended Sequence

With `dataset_source=none`, the minimal practical sequence is:

1. Train the low-level controller.
2. Convert the low-level controller checkpoint into the PyTorch checkpoint format expected by `scots_ogbench.py`.
3. Train the hierarchical diffusion planner.
4. Evaluate the diffusion planner with the low-level controller.

`scots_ogbench.py` does not train the low-level controller. It only trains the two diffusion planner components.

## GCBC Workflow

GCBC support has been added to this repo. The relevant files are:

- `scripts/low_level_controller/agents/gcbc.py`
- `scripts/low_level_controller/gcbc.sh`
- `scripts/low_level_controller/gcbc_jax2torch.py`
- `scripts/low_level_controller/gcbc_jax2torch.sh`
- `cleandiffuser_ex/gcbc/gcbc.py`
- `cleandiffuser_ex/gcbc/gcbc_utils.py`

`scots_ogbench.py` can now load `low_controller: gcbc` from `results/GCBC/<env>/gcbc_ckpt_latest.pt`.

## Example Commands

### antmaze-medium-stitch-v0 with GCBC

1. Train the low-level GCBC controller:

```bash
cd /Users/avenugo2/aravind/scots/scripts/low_level_controller
MUJOCO_GL=egl python main.py --env_name=antmaze-medium-stitch-v0 --agent=agents/gcbc.py --eval_episodes=50 --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5
```

2. Convert the JAX checkpoint to the PyTorch checkpoint used by `scots`:

```bash
cd /Users/avenugo2/aravind/scots/scripts/low_level_controller
MUJOCO_GL=egl python gcbc_jax2torch.py --env_name antmaze-medium-stitch-v0 --restore_epoch 1000000 --save_dir ../../results/GCBC/antmaze-medium-stitch-v0
```

3. Train the hierarchical diffusion planner on the original OGBench dataset only:

```bash
cd /Users/avenugo2/aravind/scots
MUJOCO_GL=egl python pipelines/scots/scots_ogbench.py task=antmaze-medium-stitch-v0 dataset_source=none
```

4. Evaluate the planner using the GCBC low-level controller:

```bash
cd /Users/avenugo2/aravind/scots
MUJOCO_GL=egl python pipelines/scots/scots_ogbench.py task=antmaze-medium-stitch-v0 dataset_source=none mode=inference task.low_controller=gcbc
```

### antmaze-medium-navigate-v0 with GCBC

The task config for `antmaze-medium-navigate-v0` is now available at `configs/scots/ogbench/task/antmaze-medium-navigate-v0.yaml`.

1. Train the low-level GCBC controller:

```bash
cd /Users/avenugo2/aravind/scots/scripts/low_level_controller
MUJOCO_GL=egl python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/gcbc.py --eval_episodes=50 --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5
```

2. Convert the JAX checkpoint to the PyTorch checkpoint used by `scots`:

```bash
cd /Users/avenugo2/aravind/scots/scripts/low_level_controller
MUJOCO_GL=egl python gcbc_jax2torch.py --env_name antmaze-medium-navigate-v0 --restore_epoch 1000000 --save_dir ../../results/GCBC/antmaze-medium-navigate-v0
```

3. Train the hierarchical diffusion planner:

```bash
cd /Users/avenugo2/aravind/scots
MUJOCO_GL=egl python pipelines/scots/scots_ogbench.py task=antmaze-medium-navigate-v0 dataset_source=none
```

4. Evaluate the planner using the GCBC low-level controller:

```bash
cd /Users/avenugo2/aravind/scots
MUJOCO_GL=egl python pipelines/scots/scots_ogbench.py task=antmaze-medium-navigate-v0 dataset_source=none mode=inference task.low_controller=gcbc
```
