# Copyright (c) OpenMMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon.
try:
    import hipdnn
except Exception:
    print("INFO: Please install hipdnn.")
import torch
from torch.amp import custom_bwd, custom_fwd
from torch.autograd.function import Function, once_differentiable
import threading

_tls = threading.local()

shape_dict = {}

def build_attn_graph(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights):
        input_type = torch.float32
        graph = hipdnn.pygraph(
            name="deform_attention",
            io_data_type=hipdnn.data_type.FLOAT,
            intermediate_data_type=hipdnn.data_type.FLOAT,
            compute_data_type=hipdnn.data_type.FLOAT,
            handle=_tls.handle,
        )    

        value_graph = graph.tensor_like(value.detach())
        spatial_shapes_graph = graph.tensor_like(value_spatial_shapes.detach())
        level_start_index_graph = graph.tensor_like(value_level_start_index.detach())
        sampling_locations_graph = graph.tensor_like(sampling_locations.detach())
        attention_weights_graph = graph.tensor_like(attention_weights.detach())
        output = graph.deform_attn_fprop(
            value=value_graph,
            spatial_shapes=spatial_shapes_graph,
            level_start_index=level_start_index_graph,
            sampling_locations=sampling_locations_graph,
            attention_weights=attention_weights_graph,
            name="deform_attn_fprop",
        )
        output.set_output(True).set_data_type(hipdnn.data_type.FLOAT)

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans()
        graph.check_support()
        graph.build_plans()
        return (graph, value_graph, spatial_shapes_graph, level_start_index_graph, sampling_locations_graph, attention_weights_graph, output)

def build_attn_bwd_graph(value_gpu,
                         value_spatial_shapes,
                         value_level_start_index,
                         sampling_locations,
                         attention_weights,
                         grad_output):
    graph = hipdnn.pygraph(
        name="deform_attention_bwd",
        io_data_type=hipdnn.data_type.FLOAT,
        intermediate_data_type=hipdnn.data_type.FLOAT,
        compute_data_type=hipdnn.data_type.FLOAT,
        handle=_tls.handle,
    )

    value_graph = graph.tensor_like(value_gpu.detach())
    spatial_shapes_graph = graph.tensor_like(value_spatial_shapes.detach())
    level_start_index_graph = graph.tensor_like(value_level_start_index.detach())
    sampling_locations_graph = graph.tensor_like(sampling_locations.detach())
    attention_weights_graph = graph.tensor_like(attention_weights.detach())
    grad_output_graph = graph.tensor_like(grad_output.detach())

    grad_value, grad_sampling_loc, grad_attn_weight = graph.deform_attn_dgrad(
        value=value_graph,
        spatial_shapes=spatial_shapes_graph,
        level_start_index=level_start_index_graph,
        sampling_locations=sampling_locations_graph,
        attention_weights=attention_weights_graph,
        grad_output=grad_output_graph,
        name="deform_attn_dgrad",
    )
    grad_value.set_output(True)
    grad_sampling_loc.set_output(True)
    grad_attn_weight.set_output(True)
    
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans()
    graph.check_support()
    graph.build_plans()
    
    return (graph,
            value_graph,
            spatial_shapes_graph,
            level_start_index_graph,
            sampling_locations_graph,
            attention_weights_graph,
            grad_output_graph,
            grad_value,
            grad_sampling_loc,
            grad_attn_weight)

class MultiScaleDeformableAttnFunction(Function):
    @staticmethod
    def _hcu_forward(ctx, value: torch.Tensor, value_spatial_shapes: torch.Tensor,
                value_level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor) -> torch.Tensor:
        input_type = torch.float32
        output_gpu = torch.empty(value.size()[0], attention_weights.size()[1], value.size()[2]*value.size()[3], dtype=input_type, device="cuda")

        shape_key = ','.join(map(str, value.shape)) + ','.join(map(str, value_spatial_shapes.shape))+','.join(map(str, sampling_locations.shape))+','.join(map(str, value_level_start_index.shape))+','.join(map(str, attention_weights.shape))

        if shape_key not in shape_dict:
            if not hasattr(_tls, "handle"):
                _tls.handle = hipdnn.create_handle()
            graph, value_graph, spatial_shapes_graph, level_start_index_graph, sampling_locations_graph, attention_weights_graph, output = build_attn_graph(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
            shape_dict[shape_key] = [graph, value_graph, spatial_shapes_graph, level_start_index_graph, sampling_locations_graph, attention_weights_graph, output]
        graph, value_graph, spatial_shapes_graph, level_start_index_graph, sampling_locations_graph, attention_weights_graph, output = shape_dict[shape_key]
        variant_pack = {
            value_graph: value.data_ptr(),
            spatial_shapes_graph: value_spatial_shapes.data_ptr(),
            level_start_index_graph: value_level_start_index.data_ptr(),
            sampling_locations_graph: sampling_locations.data_ptr(),
            attention_weights_graph: attention_weights.data_ptr(),
            output: output_gpu.data_ptr(),
        }

        workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8,  device=value.device)

        graph.exec(variant_pack=variant_pack, workspace=workspace.data_ptr())

        return output_gpu

    @staticmethod
    def _hcu_backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)
        
        shape_key = 'bwd'+ \
                    ','.join(map(str, value.shape)) + \
                    ','.join(map(str, sampling_locations.shape))+ \
                    ','.join(map(str, attention_weights.shape))+ \
                    ','.join(map(str, grad_output.shape))
        
        if shape_key not in shape_dict:
            if not hasattr(_tls, "handle"):
                _tls.handle = hipdnn.create_handle()
            shape_dict[shape_key] = build_attn_bwd_graph(value,
                                                         value_spatial_shapes,
                                                         value_level_start_index,
                                                         sampling_locations,
                                                         attention_weights,
                                                         grad_output)

        (
            graph,
            value_graph,
            spatial_shapes_graph,
            level_start_index_graph,
            sampling_locations_graph,
            attention_weights_graph,
            grad_output_graph,
            grad_value_graph,
            grad_sampling_loc_graph,
            grad_attn_weight_graph
        ) = shape_dict[shape_key]

        variant_pack = {
            value_graph: value.data_ptr(),
            spatial_shapes_graph: value_spatial_shapes.data_ptr(),
            level_start_index_graph: value_level_start_index.data_ptr(),
            sampling_locations_graph: sampling_locations.data_ptr(),
            attention_weights_graph: attention_weights.data_ptr(),
            grad_output_graph: grad_output.data_ptr(),
            grad_value_graph: grad_value.data_ptr(),
            grad_sampling_loc_graph: grad_sampling_loc.data_ptr(),
            grad_attn_weight_graph: grad_attn_weight.data_ptr(),
        }

        workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device=value.device)

        graph.exec(variant_pack=variant_pack, workspace=workspace.data_ptr())

        return grad_value, grad_sampling_loc, grad_attn_weight

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(ctx,
                value: torch.Tensor,
                value_spatial_shapes: torch.Tensor,
                value_level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor) -> torch.Tensor:
        """GPU version of multi-scale deformable attention.

        Args:
            value (torch.Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (torch.Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (torch.Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (torch.Tensor): The weight of sampling points
                used when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (torch.Tensor): The step used in image to column.

        Returns:
            torch.Tensor: has shape (bs, num_queries, embed_dims)
        """

        sampling_locations = sampling_locations.type_as(value)
        attention_weights = attention_weights.type_as(value)
        output = MultiScaleDeformableAttnFunction._hcu_forward(ctx, value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (Tensor): Gradient
                of output tensor of forward.

        Returns:
             Tuple[Tensor]: Gradient
                of input tensors in forward.
        """
        grad_value, grad_sampling_loc, grad_attn_weight = MultiScaleDeformableAttnFunction._hcu_backward(
            ctx,
            grad_output.contiguous(),
        )

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight


class MultiScaleDeformableAttn(torch.nn.Module):

    def __init__(self):
        super(MultiScaleDeformableAttn, self).__init__()
        
    def forward(self,
                value: torch.Tensor,
                value_spatial_shapes: torch.Tensor,
                value_level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor):
        return MultiScaleDeformableAttnFunction.apply(value,
                                                      value_spatial_shapes,
                                                      value_level_start_index,
                                                      sampling_locations,
                                                      attention_weights)
