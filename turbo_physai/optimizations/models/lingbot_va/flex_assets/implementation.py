# Derived from PyTorch v2.10.0, commit 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0.
# Upstream file: torch/_inductor/kernel/flex/flex_attention.py
# SPDX-License-Identifier: BSD-3-Clause
# See third_party/licenses/PyTorch-2.10-BSD-3-Clause.txt and
# third_party/notices/PyTorch-2.10-NOTICE.txt for upstream notices.
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon for LingBot-VA external templates and gfx936 tuning.
"""Triton Implementation of the flex_attention Kernel"""
from __future__ import annotations
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast, Optional, TYPE_CHECKING, Union
import sympy
import torch
from torch._inductor.virtualized import V
from torch.nn.attention.flex_attention import _Backend
from torch._inductor.ir import ComputedBuffer, ExternKernel, FixedLayout, TensorBox
from torch._inductor.lowering import empty, empty_strided, lowerings, register_lowering
from torch._inductor.select_algorithm import autotune_select_algorithm, SymbolicGridFn, TritonTemplate
from torch._inductor.kernel.flex.common import build_subgraph_buffer, create_indices_fake, create_num_blocks_fake_generator, create_placeholder, freeze_irnodes, get_fwd_subgraph_outputs, infer_dense_strides, load_flex_template, maybe_realize, set_head_dim_values, SubgraphResults
from torch._inductor.kernel.flex.flex_cpu import lower_cpu
from torch._inductor.kernel.flex.flex_decoding import _use_flex_decoding, create_flex_decoding_kernel
from torch._inductor.kernel.flex.flex_flash_attention import _use_flex_flash_attention, _use_flex_flash_attention_backward, create_flex_flash_attention_backward_kernel, create_flex_flash_attention_kernel
if TYPE_CHECKING:
    from torch._inductor.template_heuristics.triton import FlexBwDConfig, FlexConfig
log = logging.getLogger(__name__)
aten = torch.ops.aten
Expr = sympy.Expr
_GFX936_BWD_ARCH = 'gfx936'
_GFX936_BWD_QUERY_SHAPE = (1, 24, 7424, 128)
_GFX936_BWD_KV_LENGTHS = (7424, 512)
_GFX936_BWD_QKV_STRIDE = (3072, 128, 3072, 1)
_GFX936_BWD_REQUIRED_KERNEL_OPTIONS: Mapping[str, Any] = {'BLOCK_M1': 16, 'BLOCK_N1': 64, 'BLOCK_M2': 64, 'BLOCK_N2': 32, 'num_warps': 4, 'num_stages': 1, 'SPARSE_Q_BLOCK_SIZE': 128, 'SPARSE_KV_BLOCK_SIZE': 128, 'QK_HEAD_DIM': 128, 'V_HEAD_DIM': 128, 'GQA_SHARED_HEADS': 1, 'PRESCALE_QK': False, 'HAS_FULL_BLOCKS': True}
_GFX936_BWD_BACKEND_OPTIONS: Mapping[str, int] = {'waves_per_eu': 4, 'kpack': 2}
_GFX936_BWD_TUNING_KEY = 'USE_GFX936_BWD_TUNING'
_GFX936_BWD_KPACK_KEY = 'HIP_BACKEND_KPACK'

def _sanitize_kernel_options_for_triton(kernel_options: dict[str, Any]) -> tuple[dict[str, Any], _Backend]:
    """We always strip quotes around str values, we only need this in lowering, so we pop it here
    to avoid passing to triton constexpr dict
    """
    sanitized = dict(kernel_options)
    backend = cast(_Backend, sanitized.pop('BACKEND', 'AUTO'))
    return (sanitized, backend)

def _guard_sequence_divisibility(seq_len_q: int | sympy.Expr, seq_len_kv: int | sympy.Expr, kernel_options: Mapping[str, Any], *, backward: bool=False) -> bool:
    """Guard sequence alignment for the current sparse-block/tile candidate.

    Return True only when ShapeEnv establishes and records the condition.
    Otherwise, the generated kernel retains sequence boundary checks. Head-dim
    checks and attention-mask semantics are independent of this condition.
    """
    query_tile_keys = ('BLOCK_M1', 'BLOCK_M2') if backward else ('BLOCK_M',)
    kv_tile_keys = ('BLOCK_N1', 'BLOCK_N2') if backward else ('BLOCK_N',)
    query_alignment = math.lcm(kernel_options['SPARSE_Q_BLOCK_SIZE'], *(kernel_options[name] for name in query_tile_keys))
    kv_alignment = math.lcm(kernel_options['SPARSE_KV_BLOCK_SIZE'], *(kernel_options[name] for name in kv_tile_keys))
    divisibility_condition = sympy.And(sympy.Eq(sympy.Mod(seq_len_q, query_alignment), 0), sympy.Eq(sympy.Mod(seq_len_kv, kv_alignment), 0))
    return bool(V.graph.sizevars.guard_or_false(divisibility_condition))

def load_flex_template(name):
    from pathlib import Path
    return (Path(__file__).parent / 'templates' / (name + '.py.jinja')).read_text()

@SymbolicGridFn
def flex_attention_grid(batch_size, q_heads, num_queries, d_model, meta, *, cdiv):
    """Heads vary fastest; the template visits Q blocks in reverse."""
    return (q_heads, batch_size, cdiv(num_queries, meta['BLOCK_M']))

def get_float32_precision():
    if (torch.backends.cuda.matmul.fp32_precision == 'ieee' if torch.backends.cuda.matmul.fp32_precision != 'none' else torch.get_float32_matmul_precision() == 'highest') or torch.version.hip or torch.mtia.is_available():
        return "'ieee'"
    else:
        return "'tf32'"
flex_attention_template = TritonTemplate(name='lingbot_flex_attention', grid=flex_attention_grid, source=load_flex_template('flex_attention') + load_flex_template('utilities') + load_flex_template('common'))

def flex_attention(query, key, value, subgraph, block_mask, scale, kernel_options: dict[str, Any], score_mod_other_buffers, mask_mod_other_buffers):
    """The main lowering for the flex_attention hop
    This can currently lower to one of 3 templates:
    1. Base Triton Template
    2. Flex Decode Triton Template
    3. Cpu specific CPP template
    """
    kernel_options = dict(kernel_options)
    kernel_options.setdefault('LINGBOT_FAST_BF16', False)
    if query.get_device().type == 'cpu':
        return lower_cpu(query, key, value, subgraph, block_mask, scale, kernel_options, score_mod_other_buffers, mask_mod_other_buffers)
    small_dqk = V.graph.sizevars.evaluate_expr(sympy.Lt(query.get_size()[-1], 16))
    small_dv = V.graph.sizevars.evaluate_expr(sympy.Lt(value.get_size()[-1], 16))
    if small_dqk or small_dv:
        raise NotImplementedError(f'NYI: embedding dimension of the query, key, and value must be at least 16 but got E={query.get_size()[-1]} and Ev={value.get_size()[-1]}')
    (_, _, kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices, q_num_blocks, q_indices, full_q_num_blocks, full_q_indices, SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE, mask_graph) = block_mask
    placeholder_inps = [create_placeholder(name, dtype, query.get_device()) for (name, dtype) in [('score', query.get_dtype()), ('b', torch.int32), ('h', torch.int32), ('m', torch.int32), ('n', torch.int32)]]
    subgraph_buffer = build_subgraph_buffer(placeholder_inps + list(score_mod_other_buffers), subgraph)
    freeze_irnodes(subgraph_buffer)
    mask_graph_placeholder_inps = [create_placeholder(name, dtype, query.get_device()) for (name, dtype) in [('b', torch.int32), ('h', torch.int32), ('m', torch.int32), ('n', torch.int32)]]
    mask_graph_buffer = build_subgraph_buffer(mask_graph_placeholder_inps + list(mask_mod_other_buffers), mask_graph)
    freeze_irnodes(mask_graph_buffer)
    (kernel_options, backend) = _sanitize_kernel_options_for_triton(kernel_options)
    kernel_options = {k: V.graph.sizevars.guard_int(v) if isinstance(v, sympy.Symbol) else v for (k, v) in kernel_options.items()}
    kernel_options.setdefault('FLOAT32_PRECISION', get_float32_precision())
    enable_gqa = V.graph.sizevars.evaluate_expr(sympy.Ne(query.get_size()[1], key.get_size()[1]))
    can_use_decode = _use_flex_decoding(query, kv_indices, value, kernel_options, enable_gqa)
    use_decode = backend == 'TRITON_DECODE' or (backend == 'AUTO' and can_use_decode)
    if backend == 'TRITON_DECODE' and (not can_use_decode):
        raise RuntimeError("BACKEND='TRITON_DECODE' was specified but flex_decoding cannot be used for this input. flex_decoding is only available for short sequence lengths with specific configurations.")
    if use_decode:
        return create_flex_decoding_kernel(query, key, value, block_mask, scale, kernel_options, subgraph_buffer, mask_graph_buffer, score_mod_other_buffers, mask_mod_other_buffers)
    (query, key, value, kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices, q_num_blocks, q_indices, full_q_num_blocks, full_q_indices) = maybe_realize([query, key, value, kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices, q_num_blocks, q_indices, full_q_num_blocks, full_q_indices])
    if _use_flex_flash_attention(subgraph, mask_graph, kernel_options, num_score_mod_placeholders=len(placeholder_inps), backend=backend):
        return create_flex_flash_attention_kernel(query, key, value, block_mask, scale, kernel_options, subgraph_buffer, mask_graph_buffer, score_mod_other_buffers, mask_mod_other_buffers, kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices, mask_graph=mask_graph, subgraph=subgraph)
    score_mod_other_buffers = maybe_realize(score_mod_other_buffers)
    mask_mod_other_buffers = maybe_realize(mask_mod_other_buffers)
    freeze_irnodes(score_mod_other_buffers)
    freeze_irnodes(mask_mod_other_buffers)
    (Bq, Hq, seq_len_q, qk_head_dim) = query.get_size()
    (Bkv, Hkv, seq_len_kv, v_head_dim) = value.get_size()
    assert V.graph.sizevars.evaluate_expr(sympy.Eq(Bq, Bkv) | sympy.Eq(Bkv, 1)), f'Bq and Bkv must broadcastable. Got Bq={Bq} and Bkv={Bkv}'
    assert V.graph.sizevars.evaluate_expr(sympy.Gt(seq_len_q, 0)), 'Query length must be greater than 0'
    assert V.graph.sizevars.evaluate_expr(sympy.Gt(seq_len_kv, 0)), 'Key length must be greater than 0'
    B = Bq
    if seq_len_q % 128 != 0 or seq_len_kv % 128 != 0:
        kernel_options.setdefault('IS_DIVISIBLE', False)
    else:
        kernel_options.setdefault('IS_DIVISIBLE', True)
    q_strides = query.get_stride()
    out_size = [B, Hq, seq_len_q, v_head_dim]
    out_strides = infer_dense_strides(out_size, q_strides)
    layout = FixedLayout(query.get_device(), query.get_dtype(), [B, Hq, seq_len_q, v_head_dim], stride=[sympy.sympify(s) for s in out_strides])
    logsumexp_shape = [B, Hq, seq_len_q]
    logsumexp = empty_strided(logsumexp_shape, None, dtype=torch.float32, device=query.get_device())
    max_scores = empty_strided(logsumexp_shape, None, dtype=torch.float32, device=query.get_device())
    kernel_options.setdefault('SM_SCALE', scale)
    gqa_shared_heads = Hq // Hkv
    kernel_options.setdefault('GQA_SHARED_HEADS', gqa_shared_heads)
    has_full_blocks = full_kv_num_blocks is not None
    kernel_options.setdefault('HAS_FULL_BLOCKS', has_full_blocks)
    if not has_full_blocks:
        (full_kv_num_blocks, full_kv_indices) = (empty(0, device=query.get_device()) for _ in range(2))
    set_head_dim_values(kernel_options, qk_head_dim, v_head_dim, V.graph.sizevars)
    choices: list[Any] = []
    dtype = query.get_dtype()
    head_dim = V.graph.sizevars.guard_int(query.get_size()[-1])
    configs: list[FlexConfig] = V.choices.get_flex_attention_fwd_configs(head_dim, dtype, query.get_device().type)
    SPARSE_KV_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_KV_BLOCK_SIZE)
    SPARSE_Q_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_Q_BLOCK_SIZE)
    original_kernel_options = kernel_options.copy()
    (num_consumer_groups, num_buffers_warp_spec) = (0, 0)
    for conf in configs:
        cur_kernel_options = original_kernel_options.copy()
        for k in list(cur_kernel_options.keys()):
            if k.startswith('fwd_'):
                v = cur_kernel_options.pop(k)
                cur_kernel_options[k[4:]] = v
            if k.startswith('bwd_'):
                cur_kernel_options.pop(k)
        cur_kernel_options.setdefault('num_stages', conf.num_stages)
        cur_kernel_options.setdefault('num_warps', conf.num_warps)
        if cur_kernel_options.get('num_consumer_groups', False):
            cur_kernel_options.setdefault('num_consumer_groups', num_consumer_groups)
            cur_kernel_options.setdefault('num_buffers_warp_spec', num_buffers_warp_spec)
        cur_kernel_options.setdefault('USE_TMA', False)
        cur_kernel_options.setdefault('BLOCK_M', conf.block_m)
        cur_kernel_options.setdefault('BLOCK_N', conf.block_n)
        cur_kernel_options.setdefault('SPARSE_Q_BLOCK_SIZE', SPARSE_Q_BLOCK_SIZE)
        cur_kernel_options.setdefault('SPARSE_KV_BLOCK_SIZE', SPARSE_KV_BLOCK_SIZE)
        if cur_kernel_options['SPARSE_KV_BLOCK_SIZE'] % cur_kernel_options['BLOCK_N'] != 0 or cur_kernel_options['SPARSE_Q_BLOCK_SIZE'] % cur_kernel_options['BLOCK_M'] != 0:
            if len(configs) == 1:
                raise ValueError(f"Q and KV block size must be divisible by BLOCK_M and BLOCK_N. We got Q_BLOCK_SIZE={cur_kernel_options['SPARSE_Q_BLOCK_SIZE']} and KV_BLOCK_SIZE={cur_kernel_options['SPARSE_KV_BLOCK_SIZE']}.")
            continue
        for attrib in ['kpack', 'matrix_instr_nonkdim', 'waves_per_eu']:
            if hasattr(conf, attrib):
                cur_kernel_options[attrib] = getattr(conf, attrib)
        cur_kernel_options['IS_DIVISIBLE'] = _guard_sequence_divisibility(seq_len_q, seq_len_kv, cur_kernel_options, backward=False)
        error = flex_attention_template.maybe_append_choice(choices=choices, input_nodes=[query, key, value, logsumexp, max_scores, kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices], layout=layout, subgraphs=[subgraph_buffer, mask_graph_buffer], mutated_inputs=[logsumexp, max_scores], call_sizes=query.get_size(), **cur_kernel_options)
        if error is not None and len(configs) == 1:
            raise error
    inputs_for_autotuning = [query, key, value, logsumexp, max_scores, kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices] + list(score_mod_other_buffers) + list(mask_mod_other_buffers)
    input_gen_fns = {5: create_num_blocks_fake_generator(kv_indices), 6: create_indices_fake, 7: create_num_blocks_fake_generator(full_kv_indices), 8: create_indices_fake}
    out = autotune_select_algorithm('flex_attention', choices, [x for x in inputs_for_autotuning if isinstance(x, torch._inductor.ir.IRNode)], layout, input_gen_fns=input_gen_fns)
    out.data.data.subgraph_inps = list(score_mod_other_buffers) + list(mask_mod_other_buffers)
    out.data.data.subgraph_outs = get_fwd_subgraph_outputs(subgraph_buffer, mask_graph_buffer)
    return (out, logsumexp, max_scores)

def _guard_gfx936_bwd_tuning(query: TensorBox, key: TensorBox, value: TensorBox, kernel_options: Mapping[str, Any]) -> bool:
    """Guard the measured gfx936 backward tuning profile.

    The performance allowlist is separate from sequence-boundary correctness.
    Symbolic shape/stride conditions are recorded through ShapeEnv; unsupported
    targets retain their original backend options. Mask contents are not part
    of this allowlist.
    """
    if not torch.version.hip or query.get_device().type != 'cuda' or any((tensor.get_dtype() != torch.bfloat16 for tensor in (query, key, value))):
        return False
    architecture = getattr(torch.cuda.get_device_properties(query.get_device()), 'gcnArchName', '')
    if architecture.split(':', 1)[0] != _GFX936_BWD_ARCH:
        return False
    if any((kernel_options.get(name) != required_value for (name, required_value) in _GFX936_BWD_REQUIRED_KERNEL_OPTIONS.items())):
        return False
    query_shape = query.get_size()
    key_shape = key.get_size()
    value_shape = value.get_size()
    shape_and_stride_conditions = [sympy.Eq(query_shape[axis], required) for (axis, required) in enumerate(_GFX936_BWD_QUERY_SHAPE)]
    shape_and_stride_conditions.append(sympy.Or(*(sympy.Eq(key_shape[2], length) for length in _GFX936_BWD_KV_LENGTHS)))
    for shape in (key_shape, value_shape):
        shape_and_stride_conditions.extend([sympy.Eq(shape[0], _GFX936_BWD_QUERY_SHAPE[0]), sympy.Eq(shape[1], _GFX936_BWD_QUERY_SHAPE[1]), sympy.Eq(shape[2], key_shape[2]), sympy.Eq(shape[3], _GFX936_BWD_QUERY_SHAPE[3])])
    for tensor in (query, key, value):
        shape_and_stride_conditions.extend((sympy.Eq(actual, required) for (actual, required) in zip(tensor.get_stride(), _GFX936_BWD_QKV_STRIDE)))
    return bool(V.graph.sizevars.guard_or_false(sympy.And(*shape_and_stride_conditions)))

@SymbolicGridFn
def flex_attention_backward_grid(batch_size, q_heads, num_queries, d_model, kv_heads, num_key_value, meta, *, cdiv):
    """Interleave KV heads with dK/dV blocks logically preceding dQ blocks."""
    num_programs = cdiv(num_key_value, meta['BLOCK_N1']) + cdiv(num_queries, meta['BLOCK_M2']) * (q_heads // kv_heads)
    return (kv_heads, batch_size, num_programs)
flex_attention_backward_template = TritonTemplate(name='lingbot_flex_attention_backward', grid=flex_attention_backward_grid, source=load_flex_template('flex_backwards') + load_flex_template('utilities'))

def validate_joint_graph(joint_graph: torch.fx.Graph):
    """We do some pre lowering graph checks in order to raise nicer error messages"""
    for node in joint_graph.nodes:
        if node.op == 'call_function' and node.target is torch.ops.flex_lib.zeros_and_scatter.default:
            for user in node.users:
                if user.op != 'output':
                    raise NotImplementedError('Using multiple indexing operations on the same tensor that requires gradients in a score_mod function is not currently supported. This typically happens when indexing the same tensor multiple times, like:\n\n    def score_mod(score, b, h, q_idx, kv_idx):\n        return score + bias[q_idx] + bias[kv_idx]  # bias used twice!\n\nA valid workaround is to clone() the tensors that will be indexed multiple times. For example:\n\n    bias1 = bias.clone()\n    def score_mod(score, b, h, q_idx, kv_idx):\n        return score + bias[q_idx] + bias1[kv_idx]\n\nNote that this solution will use additional memory.')
    return

@dataclass(frozen=True)
class JointOutputResult:
    """Results from processing joint outputs."""
    grad_input: ComputedBuffer
    captured_grads_compute: list[ComputedBuffer]
    captured_grads: list[Optional[TensorBox]]
    mutated_grads: list[TensorBox]

def process_joint_outputs(all_joint_outputs: SubgraphResults, num_placeholders: int) -> JointOutputResult:
    """Process joint outputs and extract various buffers needed for lowering

    Args:
        all_joint_outputs: List of all the outputs from build_subgraphs
        num_placeholders: The number of placeholder inputs, used to skip over unused backward compute buffers

    Returns:
        JointOutputResult containing processed buffers and gradients
    """
    assert isinstance(all_joint_outputs, list)
    assert all_joint_outputs[0] is not None, 'joint_subgraph_buffer is None - this is a bug!'
    joint_buffer = all_joint_outputs[0]
    other_grads = all_joint_outputs[num_placeholders - 1:]
    grads_compute = [buf for buf in other_grads if buf is not None]

    def get_out(buf):
        if buf is None:
            return None
        assert isinstance(buf, ComputedBuffer)
        assert buf.name is not None
        return TensorBox.create(V.graph.get_buffer(buf.name))
    grads_out = [get_out(x) for x in other_grads]
    mutated_grads = [buf for buf in grads_out if buf is not None]
    return JointOutputResult(grad_input=joint_buffer, captured_grads_compute=grads_compute, captured_grads=grads_out, mutated_grads=mutated_grads)

def flex_attention_backward(*args, **kwargs):
    """Lowering for the flex_attention_backward op in triton"""
    (query, key, value, out, logsumexp, grad_out, grad_logsumexp, fw_graph, joint_graph, block_mask, scale, kernel_options, score_mod_other_buffers, mask_mod_other_buffers) = args
    (_, _, kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices, q_num_blocks, q_indices, full_q_num_blocks, full_q_indices, SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE, mask_graph) = block_mask
    (query, key, value, logsumexp, grad_out, kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices, q_num_blocks, q_indices, full_q_num_blocks, full_q_indices) = maybe_realize([query, key, value, logsumexp, grad_out, kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices, q_num_blocks, q_indices, full_q_num_blocks, full_q_indices])
    device = query.get_device()
    dtype = query.get_dtype()
    (Bq, Hq, seq_len_q, qk_head_dim) = query.get_size()
    (Bkv, Hkv, seq_len_kv, v_head_dim) = value.get_size()
    assert V.graph.sizevars.evaluate_expr(sympy.Eq(Bq, Bkv) | sympy.Eq(Bkv, 1)), f'Bq and Bkv must broadcastable. Got Bq={Bq} and Bkv={Bkv}'
    (kernel_options, backend) = _sanitize_kernel_options_for_triton(kernel_options)
    kernel_options = {k: V.graph.sizevars.guard_int(v) if isinstance(v, sympy.Symbol) else v for (k, v) in kernel_options.items()}
    kernel_options.setdefault('FLOAT32_PRECISION', get_float32_precision())
    seq_q_divisible = V.graph.sizevars.statically_known_true(seq_len_q % 128 == 0)
    seq_kv_divisible = V.graph.sizevars.statically_known_true(seq_len_kv % 128 == 0)
    if seq_q_divisible and seq_kv_divisible:
        kernel_options.setdefault('IS_DIVISIBLE', True)
    else:
        kernel_options.setdefault('IS_DIVISIBLE', False)
    fwd_placeholder_inps = [create_placeholder(name, dtype, device) for (name, dtype) in [('score', dtype), ('b', torch.int32), ('h', torch.int32), ('m', torch.int32), ('n', torch.int32)]]
    fw_subgraph_buffer = build_subgraph_buffer(fwd_placeholder_inps + list(score_mod_other_buffers), fw_graph)
    freeze_irnodes(fw_subgraph_buffer)
    joint_placeholder_inps = fwd_placeholder_inps + [create_placeholder('grad_score_mod', dtype, device)]
    joint_graph.graph_module.graph.eliminate_dead_code()
    validate_joint_graph(joint_graph.graph_module.graph)
    all_joint_outputs = build_subgraph_buffer(joint_placeholder_inps + list(score_mod_other_buffers), joint_graph)
    freeze_irnodes(all_joint_outputs)
    joint_outputs = process_joint_outputs(all_joint_outputs, len(joint_placeholder_inps))
    mask_graph_placeholder_inps = [create_placeholder(name, dtype, query.get_device()) for (name, dtype) in [('b', torch.int32), ('h', torch.int32), ('m', torch.int32), ('n', torch.int32)]]
    mask_graph_buffer = build_subgraph_buffer(mask_graph_placeholder_inps + list(mask_mod_other_buffers), mask_graph)
    freeze_irnodes(mask_graph_buffer)
    if _use_flex_flash_attention_backward(fw_graph, mask_graph, backend=backend):
        return create_flex_flash_attention_backward_kernel(query, key, value, out, logsumexp, grad_out, scale, kernel_options)
    key_size = [Bq, Hkv, seq_len_kv, qk_head_dim]
    key_strides = infer_dense_strides(key_size, key.get_stride())
    layout_broadcasted_k = FixedLayout(key.get_device(), key.get_dtype(), key_size, stride=[sympy.sympify(s) for s in key_strides])
    grad_lse_exp2 = lowerings[aten.mul](grad_logsumexp, 1 / math.log(2))
    mul_delta = lowerings[aten.mul](out, grad_out)
    delta = lowerings[aten.sum](mul_delta, axis=-1)
    delta = lowerings[aten.sub](delta, grad_lse_exp2)
    delta = ExternKernel.require_contiguous(delta)
    (grad_lse_exp2, delta) = maybe_realize([grad_lse_exp2, delta])
    query_size = [Bq, Hq, seq_len_q, qk_head_dim]
    grad_query_strides = infer_dense_strides(query_size, query.get_stride())
    grad_query = empty_strided(query_size, stride=[sympy.sympify(s) for s in grad_query_strides], dtype=query.get_dtype(), device=query.get_device())
    value_size = [Bq, Hkv, seq_len_kv, v_head_dim]
    value_strides = infer_dense_strides(value_size, value.get_stride())
    broadcasted_grad_value = empty_strided(value_size, stride=[sympy.sympify(s) for s in value_strides], dtype=value.get_dtype(), device=value.get_device())
    kernel_options.setdefault('SM_SCALE', scale)
    gqa_shared_heads = Hq // Hkv
    kernel_options.setdefault('GQA_SHARED_HEADS', gqa_shared_heads)
    has_full_blocks = full_kv_num_blocks is not None
    kernel_options.setdefault('HAS_FULL_BLOCKS', has_full_blocks)
    if not has_full_blocks:
        (full_kv_num_blocks, full_kv_indices, full_q_num_blocks, full_q_indices) = (empty(0, device=query.get_device()) for _ in range(4))
    set_head_dim_values(kernel_options, qk_head_dim, v_head_dim, V.graph.sizevars)
    SPARSE_Q_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_Q_BLOCK_SIZE)
    SPARSE_KV_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_KV_BLOCK_SIZE)
    choices: list[Any] = []
    dtype = query.get_dtype()
    head_dim = V.graph.sizevars.guard_int(query.get_size()[-1])
    configs: list[FlexBwDConfig] = V.choices.get_flex_attention_bwd_configs(head_dim, dtype, query.get_device().type)
    (num_consumer_groups, num_buffers_warp_spec) = (0, 0)
    original_kernel_options = kernel_options.copy()
    for conf in configs:
        if SPARSE_KV_BLOCK_SIZE % conf.block_n1 != 0 or SPARSE_Q_BLOCK_SIZE % conf.block_m1 != 0 or SPARSE_KV_BLOCK_SIZE % conf.block_n2 != 0 or (SPARSE_Q_BLOCK_SIZE % conf.block_m2 != 0):
            continue
        cur_kernel_options = original_kernel_options.copy()
        for k in list(cur_kernel_options.keys()):
            if k.startswith('bwd_'):
                v = cur_kernel_options.pop(k)
                cur_kernel_options[k[4:]] = v
            if k.startswith('fwd_'):
                cur_kernel_options.pop(k)
        cur_kernel_options.setdefault('num_warps', conf.num_warps)
        cur_kernel_options.setdefault('num_stages', conf.num_stages)
        if cur_kernel_options.get('num_consumer_groups', False):
            cur_kernel_options.setdefault('num_consumer_groups', num_consumer_groups)
            cur_kernel_options.setdefault('num_buffers_warp_spec', num_buffers_warp_spec)
        cur_kernel_options.setdefault('BLOCK_M1', conf.block_m1)
        cur_kernel_options.setdefault('BLOCK_N1', conf.block_n1)
        cur_kernel_options.setdefault('BLOCK_M2', conf.block_m2)
        cur_kernel_options.setdefault('BLOCK_N2', conf.block_n2)
        cur_kernel_options.setdefault('SPARSE_Q_BLOCK_SIZE', SPARSE_Q_BLOCK_SIZE)
        cur_kernel_options.setdefault('SPARSE_KV_BLOCK_SIZE', SPARSE_KV_BLOCK_SIZE)
        for attrib in ['kpack', 'matrix_instr_nonkdim', 'waves_per_eu']:
            if hasattr(conf, attrib):
                cur_kernel_options[attrib] = getattr(conf, attrib)
        cur_kernel_options['IS_DIVISIBLE'] = _guard_sequence_divisibility(seq_len_q, seq_len_kv, cur_kernel_options, backward=True)
        use_gfx936_bwd_tuning = _guard_gfx936_bwd_tuning(query, key, value, cur_kernel_options)
        cur_kernel_options[_GFX936_BWD_TUNING_KEY] = use_gfx936_bwd_tuning
        cur_kernel_options.pop(_GFX936_BWD_KPACK_KEY, None)
        if use_gfx936_bwd_tuning:
            cur_kernel_options['waves_per_eu'] = _GFX936_BWD_BACKEND_OPTIONS['waves_per_eu']
            cur_kernel_options[_GFX936_BWD_KPACK_KEY] = _GFX936_BWD_BACKEND_OPTIONS['kpack']
        flex_attention_backward_template.maybe_append_choice(choices=choices, input_nodes=[query, key, value, logsumexp, delta, grad_out, grad_query, broadcasted_grad_value, kv_num_blocks, kv_indices, q_num_blocks, q_indices, full_kv_num_blocks, full_kv_indices, full_q_num_blocks, full_q_indices], layout=layout_broadcasted_k, subgraphs=[fw_subgraph_buffer, joint_outputs.grad_input, mask_graph_buffer, joint_outputs.captured_grads_compute], mutated_inputs=[grad_query, broadcasted_grad_value, *joint_outputs.mutated_grads], call_sizes=query.get_size() + key.get_size()[1:3], **cur_kernel_options)
    inputs_for_autotuning = [query, key, value, logsumexp, delta, grad_out, grad_query, broadcasted_grad_value, kv_num_blocks, kv_indices, q_num_blocks, q_indices, full_kv_num_blocks, full_kv_indices, full_q_num_blocks, full_q_indices] + list(score_mod_other_buffers) + list(mask_mod_other_buffers) + joint_outputs.mutated_grads
    input_gen_fns = {8: create_num_blocks_fake_generator(kv_indices), 9: create_indices_fake, 10: create_num_blocks_fake_generator(q_indices), 11: create_indices_fake, 12: create_num_blocks_fake_generator(full_kv_indices), 13: create_indices_fake, 14: create_num_blocks_fake_generator(full_q_indices), 15: create_indices_fake}
    broadcasted_grad_key = autotune_select_algorithm('flex_attention_backward', choices, [x for x in inputs_for_autotuning if isinstance(x, torch._inductor.ir.IRNode)], layout_broadcasted_k, input_gen_fns=input_gen_fns)
    broadcasted_grad_key.data.data.subgraph_inps = list(score_mod_other_buffers) + list(mask_mod_other_buffers)
    broadcasted_grad_key.data.data.subgraph_outs = get_bwd_subgraph_outputs(fw_subgraph_buffer, mask_graph_buffer, joint_outputs)
    if V.graph.sizevars.evaluate_expr(sympy.Eq(Bq, Bkv)):
        grad_key = broadcasted_grad_key
        grad_value = broadcasted_grad_value
    else:
        assert V.graph.sizevars.evaluate_expr(sympy.Gt(Bq, 1) & sympy.Eq(Bkv, 1)), f'Bq and Bkv must broadcastable. Got Bq={V.graph.sizevars.evaluate_expr(Bq)} and Bkv={V.graph.sizevars.evaluate_expr(Bkv)}'
        grad_key = lowerings[aten.sum](broadcasted_grad_key, axis=0, keepdims=True)
        grad_value = lowerings[aten.sum](broadcasted_grad_value, axis=0, keepdims=True)
    return (grad_query, grad_key, grad_value, tuple(joint_outputs.captured_grads))

def get_bwd_subgraph_outputs(subgraph_buffer: SubgraphResults, mask_graph_buffer: SubgraphResults, joint_outputs: JointOutputResult) -> list[Optional[Union[ComputedBuffer, TensorBox]]]:
    subgraph_buffer = subgraph_buffer if isinstance(subgraph_buffer, Sequence) else [subgraph_buffer]
    mask_graph_buffer = mask_graph_buffer if isinstance(mask_graph_buffer, Sequence) else [mask_graph_buffer]
    joint_output_buffers = [joint_outputs.grad_input, *joint_outputs.captured_grads_compute, *joint_outputs.captured_grads, *joint_outputs.mutated_grads]
    return [*subgraph_buffer, *mask_graph_buffer, *joint_output_buffers]
