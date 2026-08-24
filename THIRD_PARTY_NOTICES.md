# Third-Party Notices and Provenance

This file records the provenance of source code distributed in TurboPhysAI. The
root [BSD-3-Clause License](LICENSE) applies to Hygon-authored material only. A source
file derived from an upstream work remains subject to its upstream license and
notices.

## Confirmed Apache-2.0 derivations

| Component | turbo_physai paths | Upstream repository and fixed commit | Upstream paths | License | NOTICE |
| --- | --- | --- | --- | --- | --- |
| BEVFusion / OpenMMLab mmdet3d | `turbo_physai/optimizations/models/bevfusion/{backbone,depth,gaussian,training,transfusion,transfusion_bbox_coder,transfusion_bbox_coder_runtime}.py`; `turbo_physai/optimizations/common/mmdet3d/{bev_pool,gaussian,sparse_conv,sparse_tensor,voxelization}.py`; `kernel/{bev_pool,voxelization}/` | <https://github.com/mit-han-lab/bevfusion> @ `326653dc06e0938edf1aae7d01efcd158ba83de5` | `mmdet3d/{models,core,ops,apis}/**` | Apache-2.0 | No `NOTICE` file at the fixed baseline. |
| BEVFusion bundled spconv / tensorview | `kernel/sparse_conv/`, except `include/prettyprint.h` | <https://github.com/mit-han-lab/bevfusion> @ `326653dc06e0938edf1aae7d01efcd158ba83de5` | `mmdet3d/ops/spconv/**` | Apache-2.0 | No `NOTICE` file at the fixed baseline. |
| BEVFormer | `turbo_physai/optimizations/models/bevformer/{backbone,data,geometry_sca,grid_mask,mdc,msda,training,tsa}.py` | <https://github.com/fundamentalvision/BEVFormer> @ `66b65f3a1f58caf0507cb2a971b9c0e7f842376c` | `projects/mmdet3d_plugin/**` | Apache-2.0 | No `NOTICE` file at the fixed baseline. |
| MMCV | `turbo_physai/operators/{modulated_deform_conv,multi_scale_deformable_attn}.py`; `turbo_physai/optimizations/common/mmcv/msda.py`; `turbo_physai/optimizations/models/bevformer/mdc.py` | <https://github.com/open-mmlab/mmcv> @ `ccdc61c0878d27ac7cccfecd7b474320817f0bbf` (v1.4.3) | `mmcv/ops/{modulated_deform_conv,multi_scale_deform_attn}.py` | Apache-2.0 | No `NOTICE` file identified at the fixed baseline. |

The full Apache-2.0 text is in [third_party/licenses/Apache-2.0.txt](third_party/licenses/Apache-2.0.txt).

## Confirmed BSD-3-Clause derivations

| Component | turbo_physai paths | Upstream repository and fixed commit | Upstream paths | License | NOTICE |
| --- | --- | --- | --- | --- | --- |
| PyTorch | `turbo_physai/operators/{grid_sample,upsample_bilinear_2d}.py`; `kernel/grid_sample/{GridSampler.cu,GridSampler.cuh}`; `kernel/upsample_bilinear_2d/UpSampleBilinear2d.cu` | <https://github.com/pytorch/pytorch> @ `e2d141dbde55c2a4370fac5165b0561b6af4798b` (v2.7.1) | `torch/nn/functional.py` (`grid_sample`, `interpolate`); `aten/src/ATen/native/cuda/{GridSampler.cu,GridSampler.cuh,UpSampleBilinear2d.cu}` | BSD-3-Clause | No `NOTICE` file identified at the fixed baseline. |
| PyTorch | `kernel/common/utils.h` | <https://github.com/pytorch/pytorch> @ `49444c3e546bf240bed24a101e747422d1f8a0ee` (v1.13.1) | `aten/src/ATen/native/UpSample.h` | BSD-3-Clause | No `NOTICE` file identified at the fixed baseline. |

The complete PyTorch v2.7.1 license text is in [third_party/licenses/PyTorch-BSD-3-Clause.txt](third_party/licenses/PyTorch-BSD-3-Clause.txt). `UpSampleBilinear2d.cu` also retains its inherited Caffe contributor attribution.

## Confirmed MIT derivations

| Component | turbo_physai paths | Upstream repository and fixed commit | Upstream paths | License | NOTICE |
| --- | --- | --- | --- | --- | --- |
| Sparse4D | `turbo_physai/operators/deformable_aggregation.py`; `kernel/deformable_aggregation/DeformableAggregation.cu` | <https://github.com/HorizonRobotics/Sparse4D> @ `249ffbb695f4e9db628d953e2bf6d36de04bbb69` | `projects/mmdet3d_plugin/ops/deformable_aggregation.py`; `projects/mmdet3d_plugin/ops/src/deformable_aggregation_cuda.cu` | MIT | No `NOTICE` file identified at the fixed baseline. |

The complete Sparse4D license text is in [third_party/licenses/Sparse4D-MIT.txt](third_party/licenses/Sparse4D-MIT.txt).

## Confirmed Boost Software License 1.0 material

| Component | turbo_physai path | Direct distribution source and fixed commit | Upstream path | License | NOTICE |
| --- | --- | --- | --- | --- | --- |
| cxx-prettyprint | `kernel/sparse_conv/include/prettyprint.h` | <https://github.com/mit-han-lab/bevfusion> @ `326653dc06e0938edf1aae7d01efcd158ba83de5` | `mmdet3d/ops/spconv/include/prettyprint.h` | Boost-1.0 | No `NOTICE` file at the fixed baseline. |

The complete Boost Software License 1.0 text is in [third_party/licenses/Boost-1.0.txt](third_party/licenses/Boost-1.0.txt). The source file retains Louis Delacroix's original copyright and license notice.

## Provenance policy for future additions

Before public release, every newly introduced third-party or mixed-source file
must be resolved to an upstream path and 40-character commit and added here.

All previously identified source files now have a direct upstream path and a
fixed 40-character source revision. Any newly introduced third-party or mixed
source must be added to this file before public release.

## Hygon modification notice

Where Apache-2.0 source code was modified, the source file retains the upstream
copyright and license notice, then adds Hygon copyright and `Modified by
Hygon.`. The notice does not change the upstream license.
