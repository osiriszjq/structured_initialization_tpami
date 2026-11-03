# Attribution

Based on huggingface/pytorch-image-models v1.0.22, commit
78c3724f4545c0067d0b4d2f7df6337196be01ce, Apache-2.0. The upstream LICENSE,
Ross Wightman headers and component attribution (including DeiT's repeated
augmentation sampler) are retained. No upstream Git history is imported.

The structured impulse initialization is Jianqiao Zheng's work. Its original
inverse/SVD, offset/noise and V/O construction are retained in
`timm/models/structured_init.py`. The new native-base comparison and fixed PE
training coordinates are documented in README.md. Mimetic is an independent
paper-equation adapter with disclosed sinusoidal PE/CLS choices, not copied
author code or a claim of author-run reproduction.

This repository distributes code only, without datasets or pretrained weights.
