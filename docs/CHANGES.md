# Changes from timm v1.0.22

Official base commit:78c3724f4545c0067d0b4d2f7df6337196be01ce.
`upstream_blobs.json` contains the pinned official Git blobs used for verification.
`tools/check_source.py` checks every retained timm file against this inventory.

Modified existing files:

* `train.py`: add method/snapshot/diagnostic/prefix CLI flags; CPU initialization
  before optimizer/DDP, PE WD0/LR1 groups, native scaler observation, fixed epoch
  seeds/nonpersistent workers, exact unpadded rank0 evaluation with broadcast,
  full epoch-boundary recovery and selected snapshots. Native forward/backward,
  loss, augmentation and optimizer implementations remain timm's. Tail-window
  accumulation is corrected on the formal path: fixed update boundaries and
  actual remaining microbatch count, including nondivisible final windows.
* `timm/data/dataset_factory.py`: dispatch four additional explicit torchvision
  classification adapters. Existing dataset paths retain their native behavior.

Added modules:

* `timm/models/structured_init.py`: original impulse/inverse/SVD and V/O,
  paper-equation Mimetic, one-shot position entropy, PE representation,
  optimizer grouping and inference folding. Scientific primitives are retained
  from the validated implementation; public wrapper and trainer integration
  require the actual numerical acceptance documented in EXECUTION.md.
* `timm/data/structured_datasets.py`: Food101, STL10, Flowers train+val transform
  forwarding, Oxford-IIIT Pet category splits.
* `timm/utils/structured_runtime.py`: metadata, bounded full-state recovery,
  epoch RNG policy, exact evaluation and snapshots, with no autonomous research.
* `timm/utils/structured_diagnostics.py`: optional read-only TRAIN-image attention
  entropy and effective/stable rank, with RNG and training mode restored.

`timm/models/vision_transformer.py` and all attention forward implementations
remain byte-identical to upstream. Other additions are configs, tests, tools,
documentation and CI. The upstream README/citation/requirements and root ignore
rules are replaced with project-specific descriptions. Vendor package metadata
is retained; work from the complete checkout so train.py and tools are available.
