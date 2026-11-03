# Acceptance and formal execution

The implementation is statically prepared, not remotely verified. Execute on a
real GPU host, never install/train on the coordinating Mac. From a clean clone:

1. Record host/OS, driver, GPU UUID/free memory, existing job owners, Python,
   Torch/torchvision/CUDA and dependency versions. Preserve shared resources.
2. Run source checks, CPU unit tests and reference equivalence on a preserved
   source tree with an actual accepted FE config. Keep stdout/stderr and hashes.
3. Instantiate every dataset without downloading. Validate official counts,
   class and label maps, disjointness, authoritative metadata hashes, and actual
   minibatches/transforms. Check scipy for Flowers. Build a private data_roots
   JSON whose keys are CIFAR10/CIFAR100/STL10/Food101/Flowers102/OxfordIIITPets/
   ImageNet. Prepare diagnostic TRAIN image IDs/content hashes before scores.
4. From the full fixed schedule, run short single-GPU prefixes for all four
   methods/Tiny and Small, inspect actual AMP update counts and tail windows,
   exercise save/new-process resume and compare with uninterrupted continuation.
   Use --stop-after-epochs without shortening --epochs. Preserve all failures.
5. When four GPUs are genuinely available, accept all four methods under DDP:
   same initialized tensors, PE optimizer membership/buffers, global batch1024,
   accumulation4, repeated-augmentation3 sample counts/set_epoch, AMP,
   unpadded rank0 evaluation, fold and full-state continuation. Do not replace
   this with merely showing that a torchrun process starts.
6. Measure initialization, complete train/evaluation epoch, save/diagnostic
   overhead and peak memory for each dataset/spec. Fixed seeds/recipes do not
   change based on endpoint scores. Estimate full end-to-end GPUh, and establish
   a NEW task owner, concrete task rows and measured immutable budget before
   full runs. Do not reuse historical study ledgers or budgets.
7. Run small and ImageNet suites from that exact clean commit; idle GPUs only.
   The runner reserves per-run measured GPUh, checks actual spending and retains
   failed records. If it crashes, do not erase 'running' records: reconcile the
   actual processes, owner and elapsed GPUh before changing their state.

The runner's acceptance JSON must contain:

* status=PASS, source_commit, SHA256 of the exact data_roots JSON;
* checks mapping initialization_equivalence/optimizer/full_state_resume/amp/
  tail_accumulation/data_identity/exact_evaluation to PASS, each backed by logs;
* for ImageNet, additional ddp4/ra_sampler/global_batch1024;
* estimated_full_run_gpu_hours mapping dataset/Tiny and dataset/Small to measured
  conservative full-run costs (including init/eval/save, not only training steps).

These fields are a machine gate, not proof by themselves. Attach raw commands,
resolved configs, code/environment/data hashes, host/GPU UUIDs, numerical
comparisons, actual step/skip counts, timing and complete failure logs. Never
write PASS before running the corresponding test. Return a compact manifest
and evidence archive without images, credentials or environment directories.

Formal outputs select fixed scheduled last epochs299/309, with all seeds retained.
Collect final weights+SHA and final metrics, not model-best scores. Graphs and
further component studies are deferred; representative snapshots and rank
records keep those future analyses possible. Diagnostic image manifests have
schema {partition:"train",root:"...",samples:[{path:"...",sha256:"..."}]}.
The optional suite diagnostic-manifest argument points to a mapping from dataset
name to that dataset's manifest path. Keep dataset image content private.

Full state is supported only at epoch boundaries, same world size/recipe/source.
Do not resume from folded final weights or change GPU mapping/recipe silently.
No claim of cross-platform bitwise equivalence is made.
