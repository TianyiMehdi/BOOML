# BOOM-L

Code for *Langevin Rollout Optimization for Modelic Reinforcement Learning* (ICML 2026).

BOOM-L combines Langevin Rollout Optimization (LaRO) and MPPI through Maximum Look-Ahead Planning (MLAP). The score network predicts the negative look-ahead return gradient, which LaRO subtracts to ascend the return. It is trained with a separate Adam optimizer in this implementation.

## Setup

```bash
conda activate booml
pip install -e .
pip install -r requirement.txt
```

## Train and evaluate

```bash
./run_train.sh
TASK=humanoid_h1hand-run-v0 SEED=1 ./run_train.sh
TASK=dog-run EXTRA_ARGS="save_agent=true" ./run_train.sh
python -m booml.evaluate task=dog-run exp_name=booml checkpoint=logs/dog-run/2/booml/models/final.pt
```

The script defaults to `exp_name=booml`, `task=dog-run`, and local logging. Set `EXP_NAME=boom` for the BOOM baseline or `EXP_NAME=adaptive` for adaptive LaRO. Set `DISABLE_WANDB=false` and configure WandB credentials to enable online logging.

## Main benchmark tasks

| DMC | Humanoid Bench |
| --- | --- |
| `humanoid-stand` | `humanoid_h1hand-run-v0` |
| `humanoid-walk` | `humanoid_h1hand-slide-v0` |
| `humanoid-run` | `humanoid_h1hand-pole-v0` |
| `dog-stand` | `humanoid_h1hand-stand-v0` |
| `dog-walk` | `humanoid_h1hand-walk-v0` |
| `dog-trot` | `humanoid_h1hand-sit_simple-v0` |
| `dog-run` | `humanoid_h1hand-hurdle-v0` |
