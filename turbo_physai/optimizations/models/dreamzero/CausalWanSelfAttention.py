"""Replace CausalWanSelfAttention.forward with a version whose QKV projection
and RoPE application run through torch.compile'd, tensor-only regions
(baseline model commit ab790c198fbce33503358efbbd4187ce9a89adf3).

The stock implementation calls complex-valued RoPE helpers
(rope_action_apply/causal_rope_action_apply, built on torch.polar) directly
inside forward, which causes Inductor complex-operator fallbacks. This file keeps the exact teacher
forcing / kv-cache control flow and uses real-valued RoPE (see real_rope.py).
The optimization Group can compile the QKV projection and rotation helpers.

CausalWanModel's rope_params(...) buffers (self.freqs / freqs_action /
freqs_state) must be recomputed in the real cos/sin layout for this to be
numerically consistent; see CausalWanModel.compile_real_rope_freqs.
"""

from __future__ import annotations

import functools
import os

import torch

from .real_rope import rope_qk_apply_real as _apply_rope_qk

ENABLE_TENSORRT = os.getenv("ENABLE_TENSORRT", "False").lower() == "true"

_COMPILE_DEFAULTS = {"fullgraph": True, "dynamic": True}


def _project_qkv(self, x: torch.Tensor):
    b, s = x.shape[:2]
    q = self.norm_q(self.q(x)).view(b, s, self.num_heads, self.head_dim)
    k = self.norm_k(self.k(x)).view(b, s, self.num_heads, self.head_dim)
    v = self.v(x).view(b, s, self.num_heads, self.head_dim)
    return q, k, v


def compile_forward(original, options):
    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
        return forward

    compile_kwargs = {**_COMPILE_DEFAULTS, **(options or {})}
    project_qkv = torch.compile(_project_qkv, **compile_kwargs)
    apply_rope_qk = torch.compile(_apply_rope_qk, **compile_kwargs)

    @functools.wraps(original)
    def compiled_forward(self, *args, **kwargs):
        return forward(
            self, *args, **kwargs, project_qkv=project_qkv, apply_rope_qk=apply_rope_qk,
        )

    return compiled_forward


def _prepare_rope_frequencies(
    self, freqs, freqs_action, freqs_state, action_register_length,
    action_state_index=None,
):
    # Register counts and cache positions stay outside the compiled region.
    if action_register_length is None:
        return freqs
    tokens_per_block = self.num_action_per_block + self.num_state_per_block
    if action_state_index is not None:
        assert action_register_length == tokens_per_block
    start = 0 if action_state_index is None else action_state_index
    blocks = action_register_length // tokens_per_block
    action_freqs = freqs_action[
        start * self.num_action_per_block:(start + blocks) * self.num_action_per_block
    ].reshape(blocks * self.num_action_per_block, 1, -1)
    state_freqs = freqs_state[
        start * self.num_state_per_block:(start + blocks) * self.num_state_per_block
    ].reshape(blocks * self.num_state_per_block, 1, -1)
    return torch.cat((freqs, action_freqs, state_freqs), dim=0)


def _rotate_qk(
    self, q, k, v, freqs, freqs_action, freqs_state, action_register_length,
    action_state_index=None, *, apply_rope_qk=_apply_rope_qk,
):
    if ENABLE_TENSORRT:
        from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
            causal_rope_action_apply,
            rope_action_apply,
        )
        kwargs = dict(
            freqs=freqs, freqs_action=freqs_action, freqs_state=freqs_state,
            action_register_length=action_register_length,
            num_action_per_block=self.num_action_per_block,
            num_state_per_block=self.num_state_per_block,
        )
        rotate = rope_action_apply
        if action_state_index is not None:
            rotate = causal_rope_action_apply
            kwargs["action_state_index"] = action_state_index
        return rotate(q, **kwargs).type_as(v), rotate(k, **kwargs).type_as(v)
    rotary_freqs = _prepare_rope_frequencies(
        self, freqs, freqs_action, freqs_state, action_register_length, action_state_index,
    )
    return apply_rope_qk(q, k, rotary_freqs, v.dtype)


def _rotate_teacher_forcing_qk(
    self, q, k, v, freqs, freqs_action, freqs_state, action_register_length,
    *, apply_rope_qk=_apply_rope_qk,
):
    clean_length = (q.shape[1] - (action_register_length or 0)) // 2
    if ENABLE_TENSORRT:
        clean_q, clean_k = _rotate_qk(
            self, q[:, :clean_length], k[:, :clean_length], v, freqs,
            freqs_action, freqs_state, None,
        )
        noisy_q, noisy_k = _rotate_qk(
            self, q[:, clean_length:], k[:, clean_length:], v, freqs,
            freqs_action, freqs_state, action_register_length,
        )
        return clean_q, clean_k, noisy_q, noisy_k
    noisy_freqs = _prepare_rope_frequencies(
        self, freqs, freqs_action, freqs_state, action_register_length,
    )
    # Q/K already contain clean + noisy tokens: duplicate only the small
    # frequency table, not the activations, to rotate all four slices at once.
    rotary_freqs = torch.cat((freqs, noisy_freqs), dim=0)
    rotated_q, rotated_k = apply_rope_qk(q, k, rotary_freqs, v.dtype)
    return (
        rotated_q[:, :clean_length], rotated_k[:, :clean_length],
        rotated_q[:, clean_length:], rotated_k[:, clean_length:],
    )


def process_state_blocks(self, state_q, state_k, state_v, state_horizon):
    r"""Fold independent state blocks into the batch dim for one attention call.

    State blocks only attend within themselves (no cross-block attention), so
    instead of looping over blocks and calling `self.attn` once per block,
    reshape all blocks into one larger batch and issue a single attention
    call.
    """
    if state_horizon == 0:
        return state_q

    if state_horizon % self.num_state_per_block != 0:
        raise ValueError(
            "state_horizon must be divisible by num_state_per_block, "
            f"got {state_horizon=} and num_state_per_block={self.num_state_per_block}"
        )

    num_blocks = state_horizon // self.num_state_per_block

    if num_blocks == 1:
        # Single block - one attention call
        return self.attn(state_q, state_k, state_v)

    batch_size, _, num_heads, head_dim = state_q.shape
    block_shape = (batch_size * num_blocks, self.num_state_per_block, num_heads, head_dim)

    q_blocks = state_q.reshape(block_shape)
    k_blocks = state_k.reshape(block_shape)
    v_blocks = state_v.reshape(block_shape)
    output = self.attn(q_blocks, k_blocks, v_blocks)
    return output.reshape(batch_size, state_horizon, num_heads, head_dim)


def _process_noisy_image_action_blocks(
    self, noisy_image_q, noisy_image_k, noisy_image_v,
    noisy_action_q, noisy_action_k, noisy_action_v,
    clean_image_k, clean_image_v,
    noisy_state_k, noisy_state_v,
    half_frames,
):
    """Process noisy image and action blocks with their shared K/V context."""
    block_size = self.frame_seqlen * self.num_frame_per_block
    num_blocks = (half_frames - 1) // self.num_frame_per_block

    # Pre-allocate output tensors
    image_output = torch.empty_like(noisy_image_q)
    action_output = torch.empty_like(noisy_action_q)

    # First noisy frame: conditioning, self-attention only
    image_output[:, :self.frame_seqlen] = self.attn(
        noisy_image_q[:, :self.frame_seqlen],
        noisy_image_k[:, :self.frame_seqlen],
        noisy_image_v[:, :self.frame_seqlen]
    )

    if num_blocks == 0:
        return image_output, action_output

    # Pre-compute all block indices
    noisy_block_starts = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
    noisy_block_ends = [min(start + block_size, noisy_image_q.shape[1]) for start in noisy_block_starts]
    clean_context_ends = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
    action_block_starts = [i * self.num_action_per_block for i in range(num_blocks)]
    action_block_ends = [start + self.num_action_per_block for start in action_block_starts]
    state_block_starts = [i * self.num_state_per_block for i in range(num_blocks)]
    state_block_ends = [start + self.num_state_per_block for start in state_block_starts]

    # Image and action queries use the same K/V context for each block
    for block_idx in range(num_blocks):
        noisy_start = noisy_block_starts[block_idx]
        noisy_end = noisy_block_ends[block_idx]
        clean_end = clean_context_ends[block_idx]
        action_start = action_block_starts[block_idx]
        action_end = action_block_ends[block_idx]
        state_start = state_block_starts[block_idx]
        state_end = state_block_ends[block_idx]

        image_query_length = noisy_end - noisy_start
        q_block = torch.cat([
            noisy_image_q[:, noisy_start:noisy_end],
            noisy_action_q[:, action_start:action_end]
        ], dim=1)
        k_context = torch.cat([
            clean_image_k[:, :clean_end],
            noisy_image_k[:, noisy_start:noisy_end],
            noisy_action_k[:, action_start:action_end],
            noisy_state_k[:, state_start:state_end]
        ], dim=1)
        v_context = torch.cat([
            clean_image_v[:, :clean_end],
            noisy_image_v[:, noisy_start:noisy_end],
            noisy_action_v[:, action_start:action_end],
            noisy_state_v[:, state_start:state_end]
        ], dim=1)

        block_output = self.attn(q_block, k_context, v_context)
        image_output[:, noisy_start:noisy_end] = block_output[:, :image_query_length]
        action_output[:, action_start:action_end] = block_output[:, image_query_length:]

    return image_output, action_output


def forward(
    self,
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    kv_cache: torch.Tensor | None = None,
    current_start_frame: int = 0,
    is_tf: bool = True,
    *,
    project_qkv=_project_qkv,
    apply_rope_qk=_apply_rope_qk,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Args:
        x(Tensor): Shape [B, L, num_heads, C / num_heads]
        seq_lens(Tensor): Shape [B]
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    q, k, v = project_qkv(self, x)

    updated_kv_cache: torch.Tensor | None = None

    if kv_cache is None:
        if is_tf:
            rq_context, rk_context, rq_noisy, rk_noisy = _rotate_teacher_forcing_qk(
                self, q, k, v, freqs, freqs_action, freqs_state, action_register_length,
                apply_rope_qk=apply_rope_qk,
            )

            # Calculate sequence dimensions
            half_seq_len = (s - (action_register_length if action_register_length is not None else 0)) // 2

            if action_register_length is not None:
                # Teacher forcing structure:
                # Clean half: [image tokens only]
                # Noisy half: [image tokens][action tokens][state tokens]
                # Causality only applies to image blocks!

                # Clean half contains ONLY image tokens
                clean_image_seq_len = half_seq_len
                clean_frames = clean_image_seq_len // self.frame_seqlen

                # Noisy half contains image + action + state tokens
                noisy_image_seq_len = half_seq_len
                noisy_frames = noisy_image_seq_len // self.frame_seqlen
                num_image_blocks = (noisy_frames - 1) // self.num_frame_per_block
                action_horizon = num_image_blocks * self.num_action_per_block
                state_horizon = num_image_blocks * self.num_state_per_block

                # Block layout must match actual register length. For 5B use 320x176 so latent frame_seqlen=55.
                roped_sequence_length = rq_context.shape[1] + rq_noisy.shape[1]
                if roped_sequence_length != half_seq_len + noisy_image_seq_len + action_horizon + state_horizon:
                    raise ValueError(
                        "Sequence length does not match block layout. "
                        "For 5B use 320x176 (e.g. data=dreamzero/droid_relative_wan22 or image_resolution_width=320, image_resolution_height=176). "
                        f"Got noisy_frames={noisy_frames}, num_image_blocks={num_image_blocks}, "
                        f"action_register_length={action_register_length}. "
                        "Ensure (noisy_frames - 1) // num_frame_per_block >= 1 and register length equals "
                        "num_blocks * (num_action_per_block + num_state_per_block)."
                    )

                # Split clean and noisy parts
                # Clean: [image tokens only]
                clean_image_q = rq_context
                clean_image_k = rk_context
                clean_image_v = v[:, :clean_image_seq_len]

                # Noisy: [image tokens][action tokens][state tokens]
                noisy_image_q = rq_noisy[:, :noisy_image_seq_len]
                noisy_action_q = rq_noisy[:, noisy_image_seq_len:noisy_image_seq_len + action_horizon]
                noisy_state_q = rq_noisy[:, noisy_image_seq_len + action_horizon:]

                noisy_image_k = rk_noisy[:, :noisy_image_seq_len]
                noisy_action_k = rk_noisy[:, noisy_image_seq_len:noisy_image_seq_len + action_horizon]
                noisy_state_k = rk_noisy[:, noisy_image_seq_len + action_horizon:]

                noisy_image_v = v[:, half_seq_len:half_seq_len + noisy_image_seq_len]
                noisy_action_v = v[:, half_seq_len + noisy_image_seq_len:half_seq_len + noisy_image_seq_len + action_horizon]
                noisy_state_v = v[:, half_seq_len + noisy_image_seq_len + action_horizon:]

                # ========== Process CLEAN (context) image tokens ==========
                # Clean images: simple blockwise causal attention (no action/state)
                clean_image_outputs = self._process_clean_image_only(
                    clean_image_q, clean_image_k, clean_image_v, clean_frames)

                # ========== Process NOISY tokens ==========
                # Noisy image/action queries share the same K/V context per block
                noisy_image_outputs, noisy_action_outputs = _process_noisy_image_action_blocks(
                    self,
                    noisy_image_q, noisy_image_k, noisy_image_v,
                    noisy_action_q, noisy_action_k, noisy_action_v,
                    clean_image_k, clean_image_v,
                    noisy_state_k, noisy_state_v,
                    noisy_frames)

                # Noisy state blocks: self-attention only
                noisy_state_outputs = self._process_state_blocks(
                    noisy_state_q, noisy_state_k, noisy_state_v, state_horizon)

                # Concatenate all outputs in order: clean_img, noisy_img, noisy_act, noisy_state
                x = torch.cat([
                    clean_image_outputs,
                    noisy_image_outputs, noisy_action_outputs, noisy_state_outputs
                ], dim=1)
            else:
                # No action/state tokens, fall back to simple image-only teacher forcing
                half_frames = half_seq_len // self.frame_seqlen
                clean_q = rq_context
                clean_k = rk_context
                clean_v = v[:, :half_seq_len]
                noisy_q = rq_noisy
                noisy_k = rk_noisy
                noisy_v = v[:, half_seq_len:]

                # Process clean frames with blockwise causal attention
                x_clean = self._blockwise_causal_flash_attn(
                    clean_q, clean_k, clean_v, self.frame_seqlen, self.num_frame_per_block,
                    action_horizon=None, state_horizon=None,
                    num_action_per_block=None, num_state_per_block=None,
                    visualize_mask=False)

                # Process noisy frames: attend to all clean frames + themselves
                full_k = torch.cat([clean_k, noisy_k], dim=1)
                full_v = torch.cat([clean_v, noisy_v], dim=1)
                x_noisy = self.attn(noisy_q, full_k, full_v)

                x = torch.cat([x_clean, x_noisy], dim=1)

        else:
            roped_query, roped_key = _rotate_qk(
                self, q, k, v, freqs, freqs_action, freqs_state, action_register_length,
                apply_rope_qk=apply_rope_qk,
            )

            # Calculate dynamic action and state horizons
            if action_register_length is not None:
                chunk_size = action_register_length // (self.num_action_per_block + self.num_state_per_block)
                action_horizon = chunk_size * self.num_action_per_block
                state_horizon = chunk_size * self.num_state_per_block
            else:
                action_horizon = None
                state_horizon = None

            # Use blockwise causal flash attention without massive padding
            visualize = False
            x = self._blockwise_causal_flash_attn(
                roped_query, roped_key, v, self.frame_seqlen, self.num_frame_per_block,
                action_horizon=action_horizon,
                state_horizon=state_horizon,
                num_action_per_block=self.num_action_per_block if action_register_length else None,
                num_state_per_block=self.num_state_per_block if action_register_length else None,
                visualize_mask=visualize)

    else:
        action_state_index = (current_start_frame - 1) // self.num_frame_per_block

        roped_query, roped_key = _rotate_qk(
            self, q, k, v, freqs, freqs_action, freqs_state,
            action_register_length, action_state_index, apply_rope_qk=apply_rope_qk,
        )

        # split roped_query and roped_action_query (the last action_register_length tokens)
        roped_action_query: torch.Tensor | None = None
        roped_action_key: torch.Tensor | None = None
        action_v: torch.Tensor | None = None

        if action_register_length is not None:
            roped_action_query = roped_query[:, -action_register_length:]
            roped_query = roped_query[:, :-action_register_length]
            roped_action_key = roped_key[:, -action_register_length:]
            roped_key = roped_key[:, :-action_register_length]
            action_v = v[:, -action_register_length:]
            v = v[:, :-action_register_length]
            assert roped_action_query is not None
            assert roped_action_key is not None
            assert action_v is not None

        num_new_tokens = roped_query.shape[1]
        assert roped_key.shape[1] == num_new_tokens
        assert v.shape[1] == num_new_tokens

        # If we are using local attention and the current KV cache size is larger
        # than the local attention size, we need to truncate the KV cache

        updated_kv_cache = kv_cache
        updated_k = updated_kv_cache[0]
        updated_v = updated_kv_cache[1]
        # Assign new keys/values directly up to current_end
        new_k = torch.cat([updated_k, roped_key], dim=1)
        new_v = torch.cat([updated_v, v], dim=1)

        # We may need to truncate the KV cache if it's size is larger than the max attention size.
        new_k = new_k[:, -self.max_attention_size:]
        new_v = new_v[:, -self.max_attention_size:]

        if action_register_length is not None:
            x = self.attn(
                torch.cat([roped_query, roped_action_query], dim=1),
                torch.cat([new_k, roped_action_key], dim=1),
                torch.cat([new_v, action_v], dim=1),
            )
        else:
            x = self.attn(
                roped_query,
                new_k,
                new_v,
            )
        updated_kv_cache = torch.stack([new_k, new_v], dim=0)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x, updated_kv_cache
