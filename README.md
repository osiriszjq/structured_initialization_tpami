# Structured Initialization for Vision Transformers

[Paper](https://arxiv.org/abs/2505.19985) · [Conference implementation](https://github.com/osiriszjq/structured_initialization)

Jianqiao Zheng, Xueqian Li, Hemanth Saratchandran, Simon Lucey

A clean [timm](https://github.com/huggingface/pytorch-image-models) implementation
of structured initialization and its frozen **FE18/.90** extension. It retains
the impulse/inverse/SVD construction and native ViT attention forward. Training
uses the ordinary `train.py` entry point, with explicit method parameters.

**Status:** public implementation candidate. Static checks pass; target-host
numerical, full-state resume and DDP acceptance must run before formal training.
No new formal benchmark accuracy or pretrained checkpoint is claimed yet.

## Code changes

Foundation: timm **v1.0.22**, commit `78c3724f4545c0067d0b4d2f7df6337196be01ce`.
`timm/models/vision_transformer.py` is unchanged. The only modified existing
files are `train.py` and `timm/data/dataset_factory.py`; four small modules add
initialization, explicit dataset splits, run capture and optional diagnostics.
See [the complete change list](docs/CHANGES.md). No research queue framework is
required. `python3 tools/check_source.py` checks the documented timm delta.

## Methods

Choose `--init-method`:

| Value | Construction | PE training representation |
|---|---|---|
| `default` | Native timm initialization | Ordinary trainable random PE |
| `mimetic` | Paper-equation Q/K and V/O adapter | Trainable 1D sinusoidal PE |
| `structured` | Original impulse/SVD, PE18/QK7/VO1 | Ordinary directly stored18P0 |
| `F` | Original construction | Fixed18 × trainable PE base |
| `L` | Original construction | Trainable alpha initialized18 × trainable base |
| `FE` | F plus one-shot position entropy .90 | Fixed18 × trainable PE base |
| `LE` | L plus the same calibration | Trainable alpha × trainable base |

All formal comparisons use the same native timm backbone base. This is an
explicit matched comparison, rather than a claim to recover historical run
identities. Mimetic uses sinusoidal PE and disclosed CLS/index choices; it does
not share the structured methods' random PE family. The conference implementation
remains linked above for its original usage.

FE calibrates each layer using actual `norm1(18P0)`, Q/K biases and attention
scale. One positive multiplier scales that layer's Q/K weights and biases to
mean normalized entropy .90 over heads and patch queries, keeping the CLS key.
It uses no images/labels and runs once before optimization. The multiplier is
not trained. PE base remains trainable, WD0 and LR multiplier1; the fixed18
buffer changes its training coordinates. Folded inference uses an ordinary ViT.
18/.90 are empirical frozen settings, not a proven optimum. No per-dataset tuning.

## Environment

Use Python3.11 and a compatible PyTorch/torchvision installation. The inherited
scientific implementation was observed with Torch2.13.0+cu126,
torchvision0.28.0+cu126, NumPy2.4.6 and PyYAML6.0.3. This clean implementation's
runtime remains to be verified. Reuse a compatible existing GPU environment;
do not replace another experiment's environment. `requirements.txt` lists direct
dependencies, including scipy for Flowers102. Install a CUDA-compatible PyTorch
build using the [official instructions](https://pytorch.org/get-started/locally/)
when constructing a separate new environment. No weights or dataset downloads
are performed automatically. Run from this checkout so its vendored timm is used.

```bash
python3 tools/check_source.py
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

The second command performs remote CPU numerical checks, without training or
image access. `tools/verify_reference.py` optionally compares F/L/FE/LE bitwise
against a preserved scientific source and accepted configuration; those private
inputs are not redistributed. Actual GPU/data/DDP acceptance is described in
[execution and evidence](docs/EXECUTION.md).

## Dataset preparation

Provide legally obtained local data. The `torch/...` adapters never infer a
hidden holdout or add unlabeled examples. Dataset downloads are disabled in the
formal configurations.

| Dataset | `--dataset` | `--train-split` | `--val-split` |
|---|---|---|---|
| CIFAR10 / CIFAR100 | `torch/CIFAR10` / `torch/CIFAR100` | `train` | `test` |
| STL10 | `torch/STL10` | `train` (all labeled, no fold) | `test` |
| Food101 | `torch/Food101` | `train` | `test` |
| Flowers102 | `torch/Flowers102` | `trainval` | `test` |
| Oxford-IIIT Pet | `torch/OxfordIIITPet` | `trainval` (category) | `test` |
| ImageNet1K | empty / folder dataset | `train` | `val` |

Use torchvision's expected layout for each small dataset. ImageNet uses
`ROOT/train/CLASS_ID/image` and `ROOT/val/CLASS_ID/image`, with the same class map
in both splits. Counts, metadata/label hashes, overlap and legal origin must be
verified on the actual host; folder names alone are insufficient.

## Training

Single-GPU Tiny, effective batch512 =64×accumulation8:

```bash
python train.py --config configs/small_data.yaml \
  --dataset torch/CIFAR100 --data-dir /path/to/cifar100 \
  --train-split train --val-split test --num-classes 100 \
  --init-method FE --seed 210 --output /path/to/results \
  --experiment CIFAR100_Tiny_FE_s210
```

ImageNet Tiny, global batch1024 =4×64×accumulation4:

```bash
torchrun --standalone --nproc_per_node=4 train.py --config configs/imagenet.yaml \
  --data-dir /path/to/imagenet --train-split train --val-split val \
  --num-classes 1000 --init-method FE --seed 210 \
  --output /path/to/results --experiment ImageNet_Tiny_FE_s210
```

Select Small with `--model-kwargs img_size=224 weight_init= embed_dim=384 depth=12 num_heads=6`.
The launcher writes model kwargs directly into resolved YAML, avoiding shell
quoting of the empty native weight-init string.

Small-data schedule:300 total epochs, warmup50, LR.002, AdamW, WD.05, AMP,
ImageNet normalization and the shared augmentation recipe in its config.
ImageNet corrected DeiT-derived schedule:300 cosine epochs plus10 cooldown,
warmup within those epochs, total310, fixed endpoint309; global batch1024,
explicit LR.001. Repeated augmentation3 does not triple the epoch length.
Recipes are fixed within each comparison. All official evaluation partitions
are used only for reporting; do not tune hyperparameters or choose an epoch
against them. Fresh seeds do not turn previously observed datasets into blind tests.

## Automatic formal suites

```bash
python tools/run_suite.py plan --suite small
python tools/run_suite.py plan --suite imagenet
python tools/run_suite.py run --suite small --data-roots /path/to/roots.json \
  --output /path/to/results --gpus 0,1,2,3 \
  --acceptance /path/to/actual_acceptance.json --max-gpu-hours NEW_MEASURED_BUDGET
```

`small` contains72 Tiny runs plus24 representative Small runs (CIFAR100/Food101).
`imagenet` contains24 Tiny/Small runs. Each uses four methods and common fresh
seeds210/211/212; seed210 completes first. No component grid or Base run starts
implicitly. The runner only uses currently idle allowed GPUs, records ownership
and actual GPUh, rejects unreviewed failed attempts, and stops at its budget.
It never terminates other users' processes. Acceptances and cost estimates must
be real, source/data-bound measurements, not template PASS labels.

## Results, checkpoints and later figures

`summary.csv`, resolved args, source/environment/data records and initial-state
metadata accompany every run. `final_metrics.json` records the scheduled final
epoch; comparisons never use the best epoch. Report mean/sample SD and paired
seed differences. No exploratory scores are copied into formal results.

During training, `latest.pth` and `previous.pth` hold full epoch-boundary recovery
states. Resume with the same config plus `--resume /path/to/latest.pth`; source,
recipe and world size must match. Native timm restores model/optimizer/scaler;
the extension restores scheduler and rank RNG. Nonpersistent workers and fixed
per-epoch seeds make the next data order reproducible. Mid-epoch resume is not claimed.

Every completed run retains `final_model.pth` (ordinary folded state). For runs
without representative snapshots, its two full recovery states are removed only
after final weights and metrics are saved and hashed. Representative seed210
CIFAR100/ImageNet runs retain full recovery state and snapshots at completed
epochs0/1/5/10/50. Final weights provide the final snapshot. Set
`--snapshot-epochs` explicitly for other prospectively selected analyses.

Optional `--diagnostic-data-manifest` identifies fixed TRAIN images by relative
path and SHA256. `--diagnostic-samples 16 --diagnostic-rank` records entropy,
CLS mass, effective/stable rank and singular spectra at snapshot times. Rank is
measured on the row-renormalized patch attention block, with FP64 SVD; entropy
keeps all keys including CLS. No diagnostic fits or recalibrates the model.
Weights allow attention maps to be regenerated later without saving all maps.

## Citation

```bibtex
@article{zheng2025structured,
  title={Structured Initialization for Vision Transformers},
  author={Zheng, Jianqiao and Li, Xueqian and Saratchandran, Hemanth and Lucey, Simon},
  journal={arXiv preprint arXiv:2505.19985},
  year={2025}
}
```

See [LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md). The underlying timm
code and its notices remain Apache-2.0. No dataset or checkpoint is bundled.
