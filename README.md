# [ICML 2026] BOOML-MBRL

This is the official implementation of **Langevin Rollout Optimization for Modelic Reinforcement Learning**.

## Installation

```bash
conda create -n booml python=3.10 -y
conda activate booml
pip install -e .
pip install -r requirement.txt
```

## Training

```bash
./run_train.sh
TASK=humanoid_h1hand-run-v0 SEED=1 ./run_train.sh
TASK=dog-run EXTRA_ARGS="save_agent=true" ./run_train.sh
```

The script defaults to BOOM-L on `dog-run`. Set `TASK`, `SEED`, and `CUDA_VISIBLE_DEVICES` as needed. Checkpoints are saved when `save_agent=true`.

## Evaluation

```bash
python -m booml.evaluate task=dog-run exp_name=booml checkpoint=/path/to/final.pt
```

## Tasks

| DMC | Humanoid Bench |
| --- | --- |
| `humanoid-stand` | `humanoid_h1hand-run-v0` |
| `humanoid-walk` | `humanoid_h1hand-slide-v0` |
| `humanoid-run` | `humanoid_h1hand-pole-v0` |
| `dog-stand` | `humanoid_h1hand-stand-v0` |
| `dog-walk` | `humanoid_h1hand-walk-v0` |
| `dog-trot` | `humanoid_h1hand-sit_simple-v0` |
| `dog-run` | `humanoid_h1hand-hurdle-v0` |

## Citation

```bibtex
@inproceedings{zhanglangevin,
  title={Langevin Rollout Optimization for Modelic Reinforcement Learning},
  author={Zhang, Tianyi and Wang, Likun and Zhan, Guojian and Zhang, Feihong and Guan, Yang and Lyu, Yao and Li, Shengbo Eben},
  booktitle={Forty-third International Conference on Machine Learning}
}
```
