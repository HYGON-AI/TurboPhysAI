# OpenWAM

本接入参考 OpenWAM `main` 的 `bfce20b575a9b34aea385030fcb00a9eaf61fcb5`，在运行时替换函数及其调用别名，模型源码不需要打补丁。使用原生 Wan 视频骨干和 ActionDiT，主要面向 `dual_system/joint_self_attn` 训练。

优化接入基线为 `1204dde1b9e9c232131fc99a3f2c6b78330e7f64`。请使用该上游基线，不要叠加已经内置同类优化的 OpenWAM 分支。随包配置记录了基线函数的 AST 身份；目前完成的是必要静态检查，尚未在训练环境验证多卡正确性、数值、收敛或性能。

## 启动

在已安装 OpenWAM 训练依赖和 TurboPhysAI 的 HCU 环境，从 OpenWAM 仓库根目录执行。保留原训练参数，只在命令前增加 Turbo：

```bash
turbo-physai run --model openwam --log-report \
  torchrun --nproc_per_node=8 scripts/train.py \
    dataloader=robotwin \
    dataloader.dataset_dir=/path/to/robotwin/dataset \
    dataloader.variant=clean_50 \
    '+dataloader.tasks=[blocks_ranking_size]' \
    training.finetune_ckpt_path=/path/to/OpenWAM-Alpha-Pretrain-Foundation-Model \
    training.num_epochs=null training.max_steps=30 training.batch_size=16 \
    training.dataset_num_workers=8 training.save_steps=100 \
    training.save_full_states_for_resume=false \
    training.output_path=/path/to/output \
    project.wandb.project=null
```

使用原有 DTK、FlashAttention、DeepSpeed、Accelerate 环境。Lightop 仅在显式设置 `lightop_norm: true` 后需要安装。没有新增强制的模型依赖。

## 优化分组

所有控制通过 Turbo Group 和 `options`，不读取 `OPENWAM_OPT_*`。

| Group | 内容 | 主要 options |
| --- | --- | --- |
| `openwam.data` | uint8 视频上传、RoboTwin 任务筛选和离线 latent 读取 | `uint8_preprocess`, `vae_latent_cache`, `vae_latent_cache_dir` |
| `openwam.compute` | Wan/ActionDiT pointwise、实数 RoPE、Norm、投影、SAC 算子边界 | `rope_real`, `pointwise_compile`, `lightop_norm`, `linear_bias2d` |
| `openwam.architecture` | 文本/本体感知 mask 接入、训练常量、latent 输入和连续 checkpoint 张量 | `constant_cache` |
| `openwam.mot` | 分段 Attention、block compile、尾层取消重算和选择性重算 | `mot_split_attn`, `compile_mot`, `checkpoint_skip_layers`, `sac_attention`, `sac_ffn_full`, `sac_gemm`, `sac_layers`, `sac_extra_ops`, `checkpoint_stats` |
| `openwam.text` | 冻结 T5 的前缀裁剪、LRU/离线缓存、文本长度与 FA2 padding plan | `text_cache`, `text_cache_size`, `text_cache_dir`, `t5_prefix`, `text_trim`, `padding_plan_cpu`, `fa2_padding`, `fa2_mask_fused` |
| `openwam.vae` | channels-last、因果缓存 layout、VAE pointwise 和 encoder compile | `vae_channels_last`, `vae_pointwise_compile`, `compile_vae` |
| `openwam.training` | ZeRO 参数、ReduceScatter、在线编码/deferred gather、fused AdamW、异步指标和诊断 | `zero_overlap`, `zero_reduce_scatter`, `online_encode`, `deferred_gather`, `fused_adamw`, `host_async`, `hw_queues`, `profiling`, `profile_light`, `trace_gc` |

默认配置启用数据/文本/layout/实数 RoPE 和 ZeRO overlap 设置；compile、SAC、通信调度改动、Lightop、fused AdamW、磁盘缓存和诊断默认关闭。这是待目标环境验证的接入配置，不代表新的最优性能组合。

`openwam.mot` 依赖 `openwam.compute`；`openwam.training` 依赖其余六组，以保证所需调用入口一起安装。Group 内各 options 可独立配置。

## 自定义 options

编辑随包 [recipe.yaml](../../turbo_physai/optimizations/models/openwam/configs/recipe.yaml) 的副本，然后生成自定义 OptimizationConfig。不要直接修改带生成凭据的 `optimization.yaml`。

例如在原 recipe 中调整：

```yaml
- id: openwam.mot
  options:
    compile_mot: true
    checkpoint_skip_layers: 12
    sac_ffn_full: true
    sac_layers: null
- id: openwam.vae
  options:
    vae_channels_last: true
    compile_vae: true
```

原生 30 层模型中，`checkpoint_skip_layers: 12` 表示末尾 12 层不 checkpoint，前 18 层仍 checkpoint。`sac_layers: N` 选择剩余 checkpoint 层中末尾 N 层；`null` 表示全部剩余 checkpoint 层。完整 SAC 需要联合视频/动作训练、启用 gradient checkpointing、关闭 checkpoint offload，且不使用 VACE。

在干净的基线仓库中生成配置：

```bash
turbo-physai optimization generate \
  --recipe /path/to/recipe.yaml \
  --repo /path/to/OpenWAM \
  --commit 1204dde1b9e9c232131fc99a3f2c6b78330e7f64 \
  --output /path/to/optimization.yaml

turbo-physai run --model openwam \
  --optimization-config /path/to/optimization.yaml \
  torchrun --nproc_per_node=8 scripts/train.py <原训练参数>
```

随包配置使用 [generate_openwam_config.py](../../tools/generate_openwam_config.py) 从同一基线源定义生成 AST 证据及标准生成凭据，生成过程不导入模型依赖、不运行模型。需要只做静态配置生成时也可使用此工具；它仅接受上述干净基线。

`zero_reduce_scatter`、`online_encode` 和 `deferred_gather` 需要原生 DeepSpeed ZeRO-2 且关闭 CPU offload/MoE。在线编码需要梯度累积为 1、DataLoader workers 大于 0，以及冻结的原生 Wan22Ti2v VAE38/T5。`host_async` 会将训练指标延后一拍记录。`hw_queues` 必须在 HIP 初始化之前设置，已有 `GPU_MAX_HW_QUEUES` 优先；默认 `0` 保持环境设置。性能测试保持 `profiling: off`；采集时使用 `rank0` 或 `all`。

## 离线缓存

在 recipe 的副本中设置 `openwam.data.options.vae_latent_cache_dir`，生成缓存期间保持 `vae_latent_cache: false`。如需同时生成 T5 缓存，设置 `openwam.text.options.text_cache: true`，并设置 `text_cache_dir` 或复用 latent 目录的 `text` 子目录。生成配置后，从 OpenWAM 仓库根目录执行：

```bash
turbo-physai run --model openwam \
  --optimization-config /path/to/cache-build-optimization.yaml \
  torchrun --nproc_per_node=8 --module \
    turbo_physai.optimizations.models.openwam.cache_vae_latents \
    dataloader=robotwin <与训练一致的数据、模型和 checkpoint 参数>
```

缓存写完后，在训练 recipe 中启用 `vae_latent_cache: true` 并重新生成配置。缓存签名校验数据窗口、相机、尺寸、权重和文本编码设置；使用训练 split 且关闭随机视频增强。生成和训练必须使用相同配置与 checkpoint。数据 worker 采用原生 Linux fork 模式以继承运行时补丁。

## 来源与许可证

[provenance.json](../../turbo_physai/optimizations/models/openwam/configs/provenance.json) 记录来源提交与函数映射。OpenWAM 及其原生 Wan 派生函数保持 Apache-2.0；新增 Turbo 接入代码为 BSD-3-Clause。详见 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。
