# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `cleandiffuser/` and project-specific extensions live in `cleandiffuser_ex/`. Experiment entrypoints are in `pipelines/` (`scots/`, `stitcher/`, `invdyn/`), with Hydra configs under `configs/` following the same split. Reproducible shell workflows live in `scripts/`, including `scripts/HILP/` and `scripts/low_level_controller/`. Static figures used by the paper are in `assets/`. There is no dedicated `tests/` directory in this repo.

## Build, Test, and Development Commands
Set up the environment with:
```bash
conda create -n scots python=3.9
conda activate scots
pip install -e .
pip install -r requirements.txt
```
Run the main experiment stages from the repository root:
```bash
bash scripts/invdyn.sh
bash scripts/stitcher.sh
bash scripts/scots.sh
bash scripts/scots_eval.sh
```
For targeted runs, invoke pipeline scripts directly, for example:
```bash
python pipelines/scots/scots_ogbench.py task=pointmaze-medium-stitch-v0
python pipelines/stitcher/stitcher_ogbench.py mode=train task=antmaze-medium-stitch-v0
python pipelines/scots/scots_ogbench.py dataset_source=none task=pointmaze-medium-stitch-v0
```

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and concise module names grouped by feature. Keep new Hydra config files aligned with current naming such as `configs/scots/ogbench/task/<env>.yaml`. Prefer small, focused helpers over large inline blocks, and preserve the current import-heavy research-code style unless you are cleaning a touched file consistently.

## Testing Guidelines
This repository currently relies on script-level verification instead of an automated unit-test suite. Before opening a change, run the smallest relevant pipeline or conversion script you touched and confirm it starts cleanly. For config changes, validate with a single-task command rather than the full batch script. If you add reusable logic, add a narrow smoke test or at minimum document the exact command used to verify it.

## Commit & Pull Request Guidelines
Git history here is sparse and informal, so use short imperative commit subjects, e.g. `Add antmaze stitcher config` or `Fix OGBench dataset loading`. Keep each commit scoped to one logical change. PRs should include: a brief problem statement, the commands you ran, affected tasks/configs, and sample outputs or screenshots when behavior changes materially.

## Configuration & Output Notes
Set `MUJOCO_GL=egl` for Mujoco-based runs, matching the provided scripts. Avoid committing generated artifacts such as experiment outputs, checkpoints, `.npz` augmented datasets, or `__pycache__/` contents.
The hierarchical planner now supports `dataset_source=none|only|concat`; the default is `none`, which trains on the original OGBench dataset without requiring SCoTS augmentation files.
