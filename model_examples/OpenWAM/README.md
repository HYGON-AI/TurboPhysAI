# OpenWAM

本接入参考 OpenWAM `main` 的 `d1ea554ba44057b766fff94c0d999fb11a217ffb`，在运行时替换函数及其调用别名，模型源码不需要打补丁。使用原生 Wan 视频骨干和 ActionDiT，主要面向 `dual_system/joint_self_attn` 训练。

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

模型优化通过 Turbo Group 和 `options` 控制，启动环境由 RuntimeConfig 配置，不读取 `OPENWAM_OPT_*`。

| Group | 内容 | 主要 options |
| --- | --- | --- |
| `openwam.data` | uint8 视频上传、RoboTwin 任务筛选和离线 latent 读取 | `uint8_preprocess`, `vae_latent_cache`, `vae_latent_cache_dir` |
| `openwam.compute` | Wan/ActionDiT pointwise、实数 RoPE、Norm、投影、SAC 算子边界 | `rope_real`, `pointwise_compile`, `lightop_norm`, `linear_bias2d` |
| `openwam.architecture` | 文本/本体感知 mask 接入、训练常量、latent 输入和连续 checkpoint 张量 | `constant_cache` |
| `openwam.mot` | 分段 Attention、block compile、尾层取消重算和选择性重算 | `mot_split_attn`, `compile_mot`, `checkpoint_skip_layers`, `sac_attention`, `sac_ffn_full`, `sac_gemm`, `sac_layers`, `sac_extra_ops` |
| `openwam.text` | 冻结 T5 的前缀裁剪、LRU/离线缓存、文本长度与 FA2 padding plan | `text_cache`, `text_cache_size`, `text_cache_dir`, `t5_prefix`, `text_trim`, `padding_plan_cpu`, `fa2_padding`, `fa2_mask_fused` |
| `openwam.vae` | channels-last、因果缓存 layout、VAE pointwise 和 encoder compile | `vae_channels_last`, `vae_pointwise_compile`, `compile_vae` |
| `openwam.training` | ZeRO 参数、ReduceScatter、在线编码/deferred gather、fused AdamW、Host 异步性能路径 | `zero_overlap`, `zero_reduce_scatter`, `zero_scaled_pack`, `zero_norm_overlap`, `online_encode`, `deferred_gather`, `fused_adamw`, `host_async`, `hw_queues` |

默认配置启用数据/文本/layout/实数 RoPE 和 ZeRO overlap 设置；compile、SAC、通信调度改动、Lightop、fused AdamW、磁盘缓存默认关闭。这是待目标环境验证的接入配置，不代表新的最优性能组合。

`openwam.mot` 依赖 `openwam.compute`；`openwam.training` 依赖其余六组，以保证所需调用入口一起安装。Group 内各 options 可独立配置。

## 最新 main 性能增量

参考 `main@d1ea554ba44057b766fff94c0d999fb11a217ffb`，新增四阶段 MoT 编译与分离 Q/输出的结构化注意力路径；保持拼接 K/V 的单份存储，避免扩大反向激活驻留。SAC 仅在实际 checkpoint 区间保留 FFN 边界，非重算层允许融合。

ZeRO-2 增加有界梯度批量入桶、owner 范围计划缓存、分区复制合并、批量范数及参数 view 复用。默认 ZeRO overlap 的 reduction bucket 调整为 100M 元素，保留显式配置。

`openwam.training` 另提供两个默认关闭的可选优化：`zero_scaled_pack` 将稀疏 owner 的 BF16 缩放与打包融合；`zero_norm_overlap` 在最终梯度累积完成后，于侧流计算分区范数。两者均要求 `zero_reduce_scatter=true`；使用 Triton 的目标环境需安装匹配版本。数值裁剪边界及多卡收益仍需目标环境验证。

在线编码在梯度累积大于 1 或无 DataLoader worker 时保留普通训练与 Deferred Gather 路径。Host 异步复用 ZeRO 已计算的设备端梯度范数，避免 getter 的标量同步。缓存支持目录迁移后按元数据匹配旧 reader key，生成与读取共用相同 key。

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

`zero_reduce_scatter`、`online_encode` 和 `deferred_gather` 需要原生 DeepSpeed ZeRO-2 且关闭 CPU offload/MoE。在线编码需要梯度累积为 1、DataLoader workers 大于 0，以及冻结的原生 Wan22Ti2v VAE38/T5。`host_async` 会将训练指标延后一拍记录。`hw_queues` 必须在 HIP 初始化之前设置，已有 `GPU_MAX_HW_QUEUES` 优先；默认 `0` 保持环境设置。Turbo 不迁移模型日志实现、profiler 采集、GC trace 或 SAC 统计逻辑。

## 报告优化组合

除默认接入配置外，随包提供三份可生成的 recipe。它们复用以上七个 Group，不增加模型侧环境开关。

| 配方 | 用途 | 关键配置 |
| --- | --- | --- |
| [recipe.online.yaml](../../turbo_physai/optimizations/models/openwam/configs/recipe.online.yaml) | 在线编码训练 | 编译、Lightop、融合 mask、FFN SAC、ReduceScatter、在线编码重叠、Fused AdamW、Host 异步、Deferred Gather；尾部 12 层不重算；不读写磁盘缓存，关闭在线文本 LRU。 |
| [recipe.cache-build.yaml](../../turbo_physai/optimizations/models/openwam/configs/recipe.cache-build.yaml) | 生成 VAE latent 与 T5 特征 | 与缓存训练保持编码设置；关闭缓存读取、MoT 编译、SAC 与训练通信调度，缓存入口直接编码。 |
| [recipe.cached.yaml](../../turbo_physai/optimizations/models/openwam/configs/recipe.cached.yaml) | 读取离线缓存训练 | 在在线组合上启用 VAE/T5 磁盘缓存，尾部 24 层不重算，其余 checkpoint 层使用 FFN SAC。 |

这些配方移植报告中的选项组合，尚未完成 Turbo 路径的 HCU 正确性与性能验证。它们要求原生 Wan22Ti2v VAE38/T5、`dual_system/joint_self_attn`、ZeRO-2、无 CPU optimizer offload、gradient accumulation=1、DataLoader workers>0、gradient checkpointing 开启且 checkpoint offload 关闭；Lightop 需要在目标环境预先安装。缓存训练还需准备完整且签名一致的缓存，并确认尾部 24 层不重算所需显存。

搭配 [runtime.optimized.yaml](../../turbo_physai/optimizations/models/openwam/configs/runtime.optimized.yaml) 使用。该 RuntimeConfig 在启动 `torchrun` 前设置坐标下降调优、单线程编译、16 个硬件队列、hipBLASLt 偏好及报告中的运行环境参数。显式选择该文件会覆盖同名环境变量；需要自定义时先复制并编辑。Python `apply()` 不加载 RuntimeConfig，需要调用方在导入 Torch 前设置对应环境。

OPT-01 的 hipBLASLt 库包与配套数据目录需要由目标容器提供。`TORCH_BLAS_PREFER_HIPBLASLT=1` 只选择 backend，不会安装或切换库版本；目标环境应预先配置匹配的动态库搜索路径。

在线组合：先复制 `recipe.online.yaml`，再按上一节生成配置，启动时同时指定优化配置与运行环境配置：

```bash
turbo-physai run --model openwam --log-report \
  --optimization-config /path/to/optimization.online.yaml \
  --runtime-config /path/to/runtime.optimized.yaml \
  torchrun --nproc_per_node=8 scripts/train.py \
    training.zero_stage=2 training.offload_optimizer_device=none \
    training.gradient_accumulation_steps=1 training.dataset_num_workers=8 \
    training.use_gradient_checkpointing=true \
    training.use_gradient_checkpointing_offload=false <原训练参数>
```

缓存组合：复制 `recipe.cache-build.yaml` 和 `recipe.cached.yaml`，把两份中的 `vae_latent_cache_dir` 改成同一实际目录，再分别生成 OptimizationConfig。默认 T5 文件位于该目录的 `text` 子目录。先用生成配方执行下面的缓存入口，完成后将训练命令的优化配置换成 `optimization.cached.yaml`。两阶段使用相同数据窗口、模型与 checkpoint 参数；不要直接运行带 `/path/to/openwam-cache` 占位路径的配置。

| 报告项 | Turbo 对应位置 |
| --- | --- |
| OPT-01 hipBLASLt 迁移 | 外部库包及动态库路径；RuntimeConfig 设置 backend 偏好。 |
| OPT-02 VAE layout 与局部编译 | `openwam.vae`。 |
| OPT-03 分块注意力、FA2 varlen、CPU padding、融合 mask | `openwam.mot`、`openwam.text`、`openwam.compute`。 |
| OPT-04 全图编译、实数 RoPE、Lightop Norm | `openwam.compute`、`openwam.mot`、`openwam.vae`。 |
| OPT-05 尾部 12 层不重算与 FFN SAC | 在线配方的 `openwam.mot`。 |
| OPT-06 常量、前缀、文本裁剪、uint8 | `openwam.architecture`、`openwam.text`、`openwam.data`。 |
| OPT-07 ReduceScatter、硬件队列、在线编码重叠 | `openwam.training` 和优化 RuntimeConfig。 |
| OPT-08 离线 VAE 缓存 | `openwam.data`、`openwam.architecture`、`openwam.training` 与缓存生成入口。 |
| OPT-09 离线 T5 缓存 | `openwam.text`、`openwam.training` 与缓存生成入口。 |
| OPT-10 Fused AdamW | `openwam.training.fused_adamw`。 |
| OPT-11 坐标下降调优与 Host 异步 | 优化 RuntimeConfig 与 `openwam.training.host_async`。 |
| OPT-12 缓存模式尾部 24 层不重算 | 缓存训练配方的 `openwam.mot.checkpoint_skip_layers=24`。 |
| OPT-13 Deferred Gather | `openwam.training` 提交与清理、`openwam.mot` / `openwam.architecture` 等待边界。 |

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
