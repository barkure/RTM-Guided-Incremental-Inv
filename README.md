# RTM-Guided-Incremental-Inv

Code for **Reverse-Time-Migration-Guided Deep Incremental Inversion for Ground-Penetrating Radar**.

## Installation

Linux, Python 3.12, and [Pixi](https://pixi.sh/) are required.

```bash
pixi install
```

## Dataset

Download and extraction instructions: [data/README.md](data/README.md).

## Training

```bash
pixi run python train.py \
  --data-dir data/rock_1000 \
  --run-dir train_runs/rtm-refresh \
  --model-input-mode m0_rtm --num-stages 2 \
  --recompute-rtm-between-stages \
  --initial-model-mode train_global_constant \
  --split-seed 42 --training-seed 42 --sampler-seed 42 \
  --data-loss-weight 0 --epochs 100 --early-stopping-patience 8
```

## Evaluation

```bash
pixi run python evaluate.py \
  --checkpoint train_runs/rtm-refresh/best.pt \
  --data-dir data/rock_1000 \
  --output-dir evaluation/rtm-refresh \
  --rtm-shot-batch-size 8
```

Run either command with `--help` for more options.

## Tests

```bash
pixi run python -m unittest discover -s tests -v
```
