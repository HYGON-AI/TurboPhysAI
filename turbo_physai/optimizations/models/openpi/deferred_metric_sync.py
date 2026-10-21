"""Turbo-owned trainer-loop replacement for deferred metric synchronization."""

from __future__ import annotations


def deferred_metric_train_loop(config):
    """Run upstream training semantics while delaying loss/grad-norm D2H to logging."""

    from scripts import train_pytorch as training

    torch = training.torch
    dist = training.dist
    openpi = training.openpi
    wandb = training.wandb
    np = training.np
    jax = training.jax
    safetensors = training.safetensors

    use_ddp, local_rank, device = training.setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    training.set_seed(config.seed, local_rank)

    resuming = False
    if config.resume:
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            latest_step = training.get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                training.logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(
                    f"No valid checkpoints found in {exp_checkpoint_dir} for resume"
                )
        else:
            raise FileNotFoundError(
                f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume"
            )
    elif config.overwrite and config.checkpoint_dir.exists():
        training.shutil.rmtree(config.checkpoint_dir)
        training.logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    if not resuming:
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        training.logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        training.logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    if is_main:
        training.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    training.logging.info(
        f"Using batch size per GPU: {effective_batch_size} "
        f"(total batch size across {world_size} GPUs: {config.batch_size})"
    )
    loader, data_config = training.build_datasets(config)

    if is_main and config.wandb_enabled and not resuming:
        sample_data_loader = training._data.create_data_loader(
            config, framework="pytorch", shuffle=False
        )
        sample_batch = next(iter(sample_data_loader))
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions
        images_to_log = []
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for index in range(min(5, batch_size)):
            image = torch.cat(
                [image[index].permute(1, 2, 0) for image in sample_batch["image"].values()],
                axis=1,
            ).cpu().numpy()
            images_to_log.append(wandb.Image(image))
        wandb.log({"camera_views": images_to_log}, step=0)
        del sample_batch, observation, actions, images_to_log, image, sample_data_loader
        training.gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        training.logging.info("Cleared sample batch and data loader from memory")

    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        training.logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        training.logging.info("Gradient checkpointing is not supported for this model")

    if is_main and torch.cuda.is_available():
        training.log_memory_usage(device, 0, "after_model_creation")

    if use_ddp:
        # This replacement owns the DDP construction, so explicitly consult
        # the Turbo static-DDP dispatcher instead of relying only on a wrapper
        # around DistributedDataParallel.__init__.
        from turbo_physai.optimizations.models.openpi.ddp_state import (
            configure_optimize_ddp,
            configured_ddp_kwargs,
        )

        configure_optimize_ddp()
        ddp_kwargs = configured_ddp_kwargs(
            {
                "find_unused_parameters": True,
                "gradient_as_bucket_view": True,
                "static_graph": world_size >= 8,
            }
        )
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            **ddp_kwargs,
        )
        training.logging.info(
            "Turbo DDP runtime kwargs=%s attrs=(find_unused=%s,broadcast_buffers=%s,static_graph=%s,gradient_as_bucket_view=%s)",
            ddp_kwargs,
            getattr(model, "find_unused_parameters", None),
            getattr(model, "broadcast_buffers", None),
            getattr(getattr(model, "reducer", None), "_ddp_graph_static", None),
            getattr(model, "gradient_as_bucket_view", None),
        )

    if config.pytorch_weight_path is not None:
        training.logging.info(f"Loading weights from: {config.pytorch_weight_path}")
        model_path = training.os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(
            model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model,
            model_path,
        )
        training.logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr
    from turbo_physai.optimizations.models.openpi.fused_adamw import fused_adamw_enabled

    optimizer_kwargs = {}
    if fused_adamw_enabled():
        optimizer_kwargs["fused"] = True
        training.logging.info("Enabled fused AdamW optimizer through Turbo config")
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
        **optimizer_kwargs,
    )

    global_step = 0
    if resuming:
        global_step = training.load_checkpoint(model, optim, config.checkpoint_dir, device)
        training.logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            initial_lr = peak_lr / (warmup_steps + 1)
            return initial_lr + (peak_lr - initial_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        return end_lr + (peak_lr - end_lr) * 0.5 * (1 + np.cos(np.pi * progress))

    model.train()
    start_time = training.time.time()
    infos = []
    last_synced_loss = None
    if is_main:
        training.logging.info(
            f"Running on: {training.platform.node()} | "
            f"world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        training.logging.info(
            f"Training config: batch_size={config.batch_size}, "
            f"effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        training.logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        training.logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, "
            f"decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        training.logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, "
            f"weight_decay={config.optimizer.weight_decay}, "
            f"clip_norm={config.optimizer.clip_gradient_norm}"
        )
        training.logging.info("EMA is not supported for PyTorch training")
        training.logging.info(f"Training precision: {model_cfg.dtype}")

    # Match the reference launcher: disabling tqdm means no object creation or
    # per-step update/set_postfix calls in the replacement loop.
    disable_tqdm = training.os.environ.get("OPENPI_TURBO_DISABLE_TQDM", "0") == "1"
    pbar = (
        training.tqdm.tqdm(
            total=config.num_train_steps,
            initial=global_step,
            desc="Training",
            disable=not is_main,
        )
        if is_main and not disable_tqdm
        else None
    )

    while global_step < config.num_train_steps:
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            if global_step >= config.num_train_steps:
                break
            observation = jax.tree.map(lambda value: value.to(device), observation)
            actions = actions.to(torch.float32).to(device)
            for param_group in optim.param_groups:
                param_group["lr"] = lr_schedule(global_step)

            losses = model(observation, actions)
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)
            loss = losses.mean()
            loss.backward()

            if global_step < 5 and is_main and torch.cuda.is_available():
                training.log_memory_usage(device, global_step, "after_backward")

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.optimizer.clip_gradient_norm
            )
            optim.step()
            optim.zero_grad(set_to_none=True)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.detach_()
                    parameter.grad = None

            if is_main:
                infos.append(
                    {
                        "loss": loss.detach(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": grad_norm.detach()
                        if isinstance(grad_norm, torch.Tensor)
                        else grad_norm,
                    }
                )

            if is_main and global_step % config.log_interval == 0:
                elapsed = training.time.time() - start_time
                avg_loss = torch.stack([info["loss"] for info in infos]).mean().item()
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                last_synced_loss = avg_loss
                vals = [
                    info["grad_norm"]
                    for info in infos
                    if info.get("grad_norm") is not None
                ]
                avg_grad_norm = (
                    torch.stack(vals).mean().item()
                    if vals and isinstance(vals[0], torch.Tensor)
                    else (sum(vals) / len(vals) if vals else None)
                )
                training.logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} "
                    f"grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )
                if config.wandb_enabled and infos:
                    payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        payload["grad_norm"] = avg_grad_norm
                    wandb.log(payload, step=global_step)
                start_time = training.time.time()
                infos = []

            global_step += 1
            training.save_checkpoint(model, optim, global_step, config, is_main, data_config)
            if pbar is not None:
                pbar.update(1)
                postfix = {"lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                if last_synced_loss is not None:
                    postfix["loss"] = f"{last_synced_loss:.4f}"
                pbar.set_postfix(postfix)

    if pbar is not None:
        pbar.close()
    if is_main and config.wandb_enabled:
        wandb.finish()
    training.cleanup_ddp()
