# PointPillars TurboPhysAI integration

Target: OpenMMLab MMDetection3D at `fe25f7a51d36e3702f961e198894580d83c4387b` and its nuScenes PointPillars configuration. The model repository is not modified. The adaptations originate from the PointPillars optimization project at commit `b6e97e4adae05bf70ffe4588349f369d2af95c37`; they require independent verification on the target environment.

## Groups

| Group | Target and behavior |
| --- | --- |
| `pointpillars.assigner.tensorized_compiled` | Tensorized assignment and optional compile; set `compile: false` for the tensor-only variant. |
| `pointpillars.sampler.unique_free` | Avoid redundant `unique` after nonzero. |
| `pointpillars.iou.nearest_bev_compiled` | Compile the supported LiDAR nearest-BEV IoU path. |
| `pointpillars.voxelization.legacy_sorted` | Deterministic HCU hard voxelization; convert experimental XYZ coordinates to the model's ZYX order. |
| `pointpillars.scatter.flattened_batch` | Flatten batch scatter; duplicates fall back to the upstream implementation. |
| `pointpillars.anchor.grid_compile` | Compile the supported anchor grid path. |
| `pointpillars.geometry.limit_period` | Tensor-native `limit_period` and declared aliases. |
| `pointpillars.framework.compile_point_modules` | Invoke MMEngine's module compiler with a scoped temporary configuration. |
| `pointpillars.framework.cudnn_benchmark` | Scope backend flags to `Runner.train`. |
| `pointpillars.framework.disable_dynamo_ddp` | Scope the Dynamo DDP flag to the epoch-based training loop. |

The tensor-only and tensor-plus-compile implementations share one target and therefore use one group with an option. Dynamo's DDP flag is scoped differently from the earlier process-wide modification; equivalence at lazy compilation and distributed training time is not established. Backend settings and Dynamo settings are process-global during their wrapped calls; Engine rollback only restores declared references. Existing compiled methods and in-flight calls are not automatically reverted. The duplicate check in scatter may synchronize HCU and its performance must be measured.

The voxel replacement is limited to deterministic HCU float32 hard voxelization and falls back for unsupported inputs. XYZ is used internally; the wrapper returns `[z, y, x]`. The operator test does not establish full training loss or evaluation equivalence. The recipe is an experimental selection, not a performance or accuracy claim.

## Generate and run

Use a TurboPhysAI wheel built from the same source revision containing this directory, in an environment matching the model's dependencies. Activate the intended environment first:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv2
```

Generate the configuration using the official CLI; provide the actual installation paths:

```bash
turbo-physai optimization generate \
  --recipe /path/to/TurboPhysAI/turbo_physai/optimizations/models/PointPillars/configs/recipe.yaml \
  --repo /path/to/mmdetection3d \
  --commit fe25f7a51d36e3702f961e198894580d83c4387b \
  --output /path/to/TurboPhysAI/turbo_physai/optimizations/models/PointPillars/configs/optimization.yaml
```

Then check the generated configuration with `turbo-physai optimization check` and use the explicit config path with `turbo-physai run --optimization-config ... -- torchrun ...`. The directory name is case-sensitive; do not assume lower-case `--model` discovery selects `PointPillars`. Set dataset paths, devices and batch settings externally. The provided RuntimeConfig does not hard-code machine-specific settings.

The optional launcher is `scripts/run_pointpillars_turbo.sh`; it activates `mmcv2`, runs the official training entry through Turbo's CLI and writes outputs outside the model repository. It uses physical HCU IDs 2 and 6, batch 4/HCU and 10 workers/HCU. For example:

```bash
NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
  bash /path/to/TurboPhysAI/turbo_physai/optimizations/models/PointPillars/scripts/run_pointpillars_turbo.sh
```

The launcher accepts `DATA_ROOT`, `WORK_DIR`, `LOG_DIR`, `POINTPILLARS_REPO_ROOT` and `POINTPILLARS_CONFIG` as path overrides. Use absolute paths. The two Turbo configuration paths and `HIP_VISIBLE_DEVICES=2,6` are set directly in the script; edit them for another deployment or device selection. Additional arguments are MMEngine `KEY=VALUE` configuration overrides. A compatible Turbo wheel containing this package must be installed in the active environment and its `turbo-physai` entry point must be on `PATH`. The CLI's `--set` passes the model checkout in `PYTHONPATH` while retaining the caller's existing value. The runtime carries the earlier MIOpen and rocBLAS flags; their numerical and performance effects have not yet been validated in this configuration. It leaves network, device, NCCL topology and library paths to the caller and enables NUMA by default. NUMA requires a compatible device topology and numactl; validate them on the target deployment. The earlier short run did not load this RuntimeConfig and is not evidence for its performance or precision.

## Verification

In the `mmcv2` container, CPU contract/parity tests and a small HCU voxel comparison passed. Configuration generation/checking and an Engine entry-script check reported 10 groups applied. A single-HCU three-step smoke completed its training iterations; its unintended validation was interrupted. A two-HCU, batch-4, 10-worker, seed-42, 640-sample run completed 80 iterations. Both ranks hit Dynamo recompilation limits, so compiled coverage and steady-state behavior need further investigation. Complete accuracy, loss-trajectory, one-group-at-a-time and matched baseline throughput checks remain pending; the short run does not establish model equivalence.
