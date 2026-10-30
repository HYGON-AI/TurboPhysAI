# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Packaged DiT4DiT entry point used after TurboPhysAI applies its recipe."""

import argparse


def _set_gradient_accumulation(accelerator, accumulation):
    accumulation = int(accumulation)
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    accelerator.gradient_accumulation_steps = accumulation
    plugin = accelerator.state.deepspeed_plugin
    if plugin is None:
        raise RuntimeError("DiT4DiT requires the configured DeepSpeed plugin")
    plugin.gradient_accumulation_steps = accumulation
    plugin.deepspeed_config["gradient_accumulation_steps"] = accumulation


def main():
    from omegaconf import OmegaConf
    from DiT4DiT.training.train import accelerator, main as train
    from DiT4DiT.training.trainer_utils.trainer_tools import normalize_dotlist_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.merge(
        OmegaConf.load(args.config_yaml),
        OmegaConf.from_dotlist(normalize_dotlist_args(overrides)),
    )
    _set_gradient_accumulation(
        accelerator, cfg.trainer.gradient_accumulation_steps
    )
    train(cfg)


if __name__ == "__main__":
    main()
