# RTM-Guided-Incremental-Inv

RTM-guided deep incremental inversion for ground-penetrating radar.

Initial public code release for **Reverse-Time-Migration-Guided Deep Incremental Inversion for Ground-Penetrating Radar**.

The method uses an RTM image computed under the current relative-permittivity estimate to guide a shared residual-update network. RTM-Refresh recomputes this image after the first update; RTM-Reuse uses the same initial image for both updates. The main training objective supervises model predictions; gradients do not pass through RTM and no explicit waveform data-consistency loss is imposed in the reported main configuration.

## Release scope

This initial release contains the core implementation, training and evaluation entry points, synthetic forward-data generation, robustness evaluation, paired statistics, and selected self-contained tests. Trained weights, experiment datasets, measured profiles, complete frozen results, and gprMax experiment tooling are not included in this release. A fuller release is planned after paper acceptance. This repository does not yet provide a complete archive reproducing every reported experiment. Licensing terms will be finalized with the full release.

## Environment

Use Linux and Python 3.12 with [Pixi](https://pixi.sh/):

```bash
pixi install
pixi run python train.py --help
pixi run python evaluate.py --help
pixi run python -m unittest discover -s tests -v
```

`pixi.lock` fixes the environment; the PyTorch build uses CUDA 13.0. Training and RTM can require substantial GPU memory. Use `--device cpu` for CPU execution where practical, or a compatible CUDA GPU; shot batching can reduce RTM memory during evaluation (`--rtm-shot-batch-size`).

## Data

Dataset download and placement instructions are maintained in [`data/README.md`](data/README.md). The synthetic dataset is available through the linked iCloud Drive download. No private credentials are distributed. Samples are organized as `sample_*/data.pt` with accompanying `meta.json`; the dataset reader supports the schema produced by the generator below. Required model and observation tensors are `permittivity` and `bscan_processed`; metadata describe geometry, space/time sampling, and wavelet settings.

Generate a small synthetic dataset to inspect the format (illustrative data, not the paper dataset):

```bash
pixi run python scripts/generate_forward_dataset.py --kind layered --count 4 --output-dir data/example --device cpu
```

The forward and migration solvers use a two-dimensional lossless scalar wave model. Conductive electromagnetic observations require care because this migration operator does not model conductive losses.

## Training

For a training-derived constant background, explicitly select `train_global_constant`. The legacy default `oracle_mean` uses each target model's mean and is oracle-informed; it must not be described as an independently estimated background.

```bash
pixi run python train.py \
  --data-dir /path/to/forward_dataset \
  --run-dir train_runs/rtm-refresh \
  --model-input-mode m0_rtm --num-stages 2 \
  --recompute-rtm-between-stages \
  --initial-model-mode train_global_constant \
  --split-seed 42 --training-seed 42 --sampler-seed 42 \
  --data-loss-weight 0 --epochs 100 --early-stopping-patience 8
```

Use `--no-recompute-rtm-between-stages` for RTM-Reuse, or `--num-stages 1` for Single-stage RTM. The example command is a starting configuration; it is not a substitute for all original experiment configurations.

## Evaluation

```bash
pixi run python evaluate.py \
  --checkpoint train_runs/rtm-refresh/best.pt \
  --data-dir /path/to/forward_dataset \
  --output-dir evaluation/rtm-refresh \
  --rtm-shot-batch-size 8
```

Evaluation reconstructs the checkpoint's test split and records sample identities, per-sample metrics and summaries. Use matching sample IDs and order for paired comparisons. Physical relative-permittivity MAE and MSE are reported alongside normalized metrics; with bounds [2, 10], physical MAE is 8 times normalized MAE and physical MSE is 64 times normalized MSE. SSIM and PSNR use a fixed normalized data range. See `src/rtm_inv/protocol.py` and `src/rtm_inv/statistics.py` for definitions.

`scripts/evaluate_robustness.py` and `scripts/summarize_clean_metrics.py` provide robustness evaluation and paired statistical summaries; run them with `--help` for arguments.

## Layout

- `src/rtm_inv/`: preprocessing, RTM/forward physics, shared update network, initialization, losses, metrics and statistics.
- `train.py`, `evaluate.py`, `finetune.py`: training, clean evaluation and optional near-domain adaptation (user-supplied data).
- `scripts/`: initial public data-generation and analysis entry points.
- `tests/`: selected tests that do not require the private experiment archive.
- `pixi.toml`, `pixi.lock`: environment definition and lockfile.

The internal Python package remains named `rtm_inv` for compatibility. Only the main RTM-guided U-Net update network is included. Single-stage and fixed-RTM settings are retained as controls of this same network. External comparison networks and B-scan fusion architectures are excluded.
