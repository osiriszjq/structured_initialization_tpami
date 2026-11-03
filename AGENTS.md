# Clean timm implementation

Read README.md and docs/CHANGES.md. Standard timm v1.0.22 with documented small
extensions; no historical lab/queue framework. FE18/.90 is frozen. Never tune
alpha/entropy, replace solver/noise/V/O, or reinterpret positional entropy as
image routing quality. F/L/FE/LE are named component controls, not a search.

Mac: only light static checks; no training-stack installation or training.
Target: Fairfax4way26RTX4090 shared host, idle cards first come first served.
One execution Codex. Discover actual GPU UUIDs, processes, environment and data.
Preserve foreign jobs and active source. No sudo/reset/reboot/driver changes,
old-environment modifications, downloads needing accounts, or force pushes.

User authorized public release and formal experiment preparation/execution.
Before long runs, perform real CPU/GPU/DDP/data/resume acceptance and measure
cost; bind a new concrete budget and owner, never old research budgets. Use
train.py flags and tools/run_suite.py. No manual setup tasks for the user:
locate files, clone fresh source, discover environment/data, prepare commands.
Only ask about a concrete missing access/resource/budget after discovery.

Run a fixed clean commit. Fixes go to a separate codex/ branch/checkout, never
under active jobs. Return actual evidence and changed-file/commit provenance.
Intermediate scores do not change rows/seeds/methods or frozen recipes.

Old-server cleanup is a separate operation. First inventory ownership, active
process paths, environment/data symlink dependencies, unique checkpoints and
backups. Do not delete foreign/shared paths, live source, datasets, environments
or sole evidence. This public repository grants no blanket server deletion.
