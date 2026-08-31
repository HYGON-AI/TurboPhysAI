# Third-Party Notices and Provenance

This file records the provenance of source code distributed in TurboPhysAI. The root [BSD-3-Clause License](LICENSE) applies to Hygon-authored material only. A source file derived from an upstream work remains subject to its upstream license and notices.

## Confirmed Apache-2.0 derivations

The tables below record every local derivative file and its direct upstream source. Each source repository is fixed to a 40-character commit. `Modified` means that the upstream notice is retained and a Hygon modification notice is present in the local source. `Unmodified` means that the local file is identical to the upstream file at the fixed commit. No `NOTICE` file was present in the listed upstream baselines.

### BEVFusion / OpenMMLab mmdet3d

- Repository: <https://github.com/mit-han-lab/bevfusion>
- Fixed commit: `326653dc06e0938edf1aae7d01efcd158ba83de5`
- License: Apache-2.0

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `kernel/bev_pool/src/bev_pool_cpu.cpp` | `mmdet3d/ops/bev_pool/src/bev_pool_cpu.cpp` | Modified |
| `kernel/bev_pool/src/bev_pool_cuda.cu` | `mmdet3d/ops/bev_pool/src/bev_pool_cuda.cu` | Modified |
| `kernel/voxelization/src/scatter_points_cpu.cpp` | `mmdet3d/ops/voxel/src/scatter_points_cpu.cpp` | Modified |
| `kernel/voxelization/src/scatter_points_cuda.cu` | `mmdet3d/ops/voxel/src/scatter_points_cuda.cu` | Modified |
| `kernel/voxelization/src/voxelization.cpp` | `mmdet3d/ops/voxel/src/voxelization.cpp` | Modified |
| `kernel/voxelization/src/voxelization.h` | `mmdet3d/ops/voxel/src/voxelization.h` | Modified |
| `kernel/voxelization/src/voxelization_cpu.cpp` | `mmdet3d/ops/voxel/src/voxelization_cpu.cpp` | Modified |
| `kernel/voxelization/src/voxelization_cuda.cu` | `mmdet3d/ops/voxel/src/voxelization_cuda.cu` | Modified |
| `turbo_physai/optimizations/common/mmdet3d/bev_pool.py` | `mmdet3d/ops/bev_pool/bev_pool.py` | Modified |
| `turbo_physai/optimizations/common/mmdet3d/gaussian.py` | `mmdet3d/core/utils/gaussian.py` | Modified |
| `turbo_physai/optimizations/common/mmdet3d/sparse_conv.py` | `mmdet3d/ops/spconv/ops.py` | Modified |
| `turbo_physai/optimizations/common/mmdet3d/sparse_tensor.py` | `mmdet3d/ops/spconv/structure.py` | Modified |
| `turbo_physai/optimizations/common/mmdet3d/voxelization.py` | `mmdet3d/ops/voxel/voxelize.py` | Modified |
| `turbo_physai/optimizations/models/bevfusion/backbone.py` | `mmdet3d/models/fusion_models/bevfusion.py` | Modified |
| `turbo_physai/optimizations/models/bevfusion/depth.py` | `mmdet3d/models/vtransforms/base.py`; `mmdet3d/models/vtransforms/depth_lss.py` | Modified |
| `turbo_physai/optimizations/models/bevfusion/gaussian.py` | `mmdet3d/core/utils/gaussian.py` | Modified |
| `turbo_physai/optimizations/models/bevfusion/training.py` | `mmdet3d/models/fusion_models/base.py`; `mmdet3d/apis/train.py` | Modified |
| `turbo_physai/optimizations/models/bevfusion/transfusion.py` | `mmdet3d/models/heads/bbox/transfusion.py` | Modified |
| `turbo_physai/optimizations/models/bevfusion/transfusion_bbox_coder.py` | `mmdet3d/core/bbox/coders/transfusion_bbox_coder.py` | Modified |
| `turbo_physai/optimizations/models/bevfusion/transfusion_bbox_coder_runtime.py` | `mmdet3d/core/bbox/coders/transfusion_bbox_coder.py` | Modified |

### BEVFusion bundled spconv / tensorview

- Repository: <https://github.com/mit-han-lab/bevfusion>
- Fixed commit: `326653dc06e0938edf1aae7d01efcd158ba83de5`
- License: Apache-2.0

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `kernel/sparse_conv/include/paramsgrid.h` | `mmdet3d/ops/spconv/include/paramsgrid.h` | Modified |
| `kernel/sparse_conv/include/pybind11_utils.h` | `mmdet3d/ops/spconv/include/pybind11_utils.h` | Modified |
| `kernel/sparse_conv/include/spconv/fused_spconv_ops.h` | `mmdet3d/ops/spconv/include/spconv/fused_spconv_ops.h` | Modified |
| `kernel/sparse_conv/include/spconv/geometry.h` | `mmdet3d/ops/spconv/include/spconv/geometry.h` | Modified |
| `kernel/sparse_conv/include/spconv/indice.cu.h` | `mmdet3d/ops/spconv/include/spconv/indice.cu.h` | Modified |
| `kernel/sparse_conv/include/spconv/indice.h` | `mmdet3d/ops/spconv/include/spconv/indice.h` | Modified |
| `kernel/sparse_conv/include/spconv/maxpool.h` | `mmdet3d/ops/spconv/include/spconv/maxpool.h` | Modified |
| `kernel/sparse_conv/include/spconv/mp_helper.h` | `mmdet3d/ops/spconv/include/spconv/mp_helper.h` | Unmodified |
| `kernel/sparse_conv/include/spconv/point2voxel.h` | `mmdet3d/ops/spconv/include/spconv/point2voxel.h` | Modified |
| `kernel/sparse_conv/include/spconv/pool_ops.h` | `mmdet3d/ops/spconv/include/spconv/pool_ops.h` | Modified |
| `kernel/sparse_conv/include/spconv/reordering.cu.h` | `mmdet3d/ops/spconv/include/spconv/reordering.cu.h` | Modified |
| `kernel/sparse_conv/include/spconv/reordering.h` | `mmdet3d/ops/spconv/include/spconv/reordering.h` | Modified |
| `kernel/sparse_conv/include/spconv/spconv_ops.h` | `mmdet3d/ops/spconv/include/spconv/spconv_ops.h` | Modified |
| `kernel/sparse_conv/include/tensorview/helper_kernel.cu.h` | `mmdet3d/ops/spconv/include/tensorview/helper_kernel.cu.h` | Unmodified |
| `kernel/sparse_conv/include/tensorview/helper_launch.h` | `mmdet3d/ops/spconv/include/tensorview/helper_launch.h` | Unmodified |
| `kernel/sparse_conv/include/tensorview/tensorview.h` | `mmdet3d/ops/spconv/include/tensorview/tensorview.h` | Modified |
| `kernel/sparse_conv/include/torch_utils.h` | `mmdet3d/ops/spconv/include/torch_utils.h` | Modified |
| `kernel/sparse_conv/include/utility/timer.h` | `mmdet3d/ops/spconv/include/utility/timer.h` | Modified |
| `kernel/sparse_conv/src/all.cc` | `mmdet3d/ops/spconv/src/all.cc` | Modified |
| `kernel/sparse_conv/src/indice_cpu.cc` | `mmdet3d/ops/spconv/src/indice_cpu.cc` | Modified |
| `kernel/sparse_conv/src/indice_cuda.cu` | `mmdet3d/ops/spconv/src/indice_cuda.cu` | Modified |
| `kernel/sparse_conv/src/maxpool_cpu.cc` | `mmdet3d/ops/spconv/src/maxpool_cpu.cc` | Modified |
| `kernel/sparse_conv/src/maxpool_cuda.cu` | `mmdet3d/ops/spconv/src/maxpool_cuda.cu` | Modified |
| `kernel/sparse_conv/src/reordering_cpu.cc` | `mmdet3d/ops/spconv/src/reordering_cpu.cc` | Modified |
| `kernel/sparse_conv/src/reordering_cuda.cu` | `mmdet3d/ops/spconv/src/reordering_cuda.cu` | Modified |

The bundled `kernel/sparse_conv/include/prettyprint.h` file is registered separately under [Boost Software License 1.0 material](#confirmed-boost-software-license-10-material).

### BEVFormer

- Repository: <https://github.com/fundamentalvision/BEVFormer>
- Fixed commit: `66b65f3a1f58caf0507cb2a971b9c0e7f842376c`
- License: Apache-2.0

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `turbo_physai/optimizations/models/bevformer/backbone.py` | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` | Modified |
| `turbo_physai/optimizations/models/bevformer/data.py` | `projects/mmdet3d_plugin/datasets/builder.py` | Modified |
| `turbo_physai/optimizations/models/bevformer/geometry_sca.py` | `projects/mmdet3d_plugin/bevformer/modules/transformer.py`; `projects/mmdet3d_plugin/bevformer/modules/encoder.py`; `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py` | Modified |
| `turbo_physai/optimizations/models/bevformer/grid_mask.py` | `projects/mmdet3d_plugin/models/utils/grid_mask.py` | Modified |
| `turbo_physai/optimizations/models/bevformer/msda.py` | `projects/mmdet3d_plugin/bevformer/modules/multi_scale_deformable_attn_function.py` | Modified |
| `turbo_physai/optimizations/models/bevformer/training.py` | `projects/mmdet3d_plugin/bevformer/apis/train.py` | Modified |
| `turbo_physai/optimizations/models/bevformer/tsa.py` | `projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py` | Modified |

### MMCV

- Repository: <https://github.com/open-mmlab/mmcv>
- Fixed commit: `ccdc61c0878d27ac7cccfecd7b474320817f0bbf` (v1.4.3)
- License: Apache-2.0

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `turbo_physai/operators/modulated_deform_conv.py` | `mmcv/ops/modulated_deform_conv.py` | Modified |
| `turbo_physai/operators/multi_scale_deformable_attention.py` | `mmcv/ops/multi_scale_deform_attn.py` | Modified |
| `turbo_physai/optimizations/models/bevformer/mdc.py` | `mmcv/ops/modulated_deform_conv.py` | Modified |
| `test/msda_reference.py` | `mmcv/ops/multi_scale_deform_attn.py` | Modified |

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

## Confirmed third-party dependencies

The dependency below is installed from its published package; its source code is not copied into this repository.

| Dependency | Version | Upstream project | License | Declared in | Use |
| --- | --- | --- | --- | --- | --- |
| Pillow | `12.3.0` | <https://github.com/python-pillow/Pillow/tree/12.3.0> | MIT-CMU | `requirements.txt`; `docker/ci/Dockerfile` | Image-mask generation in the BEVFormer GridMask optimization. |

The complete Pillow 12.3.0 license text is in [third_party/licenses/Pillow-MIT-CMU.txt](third_party/licenses/Pillow-MIT-CMU.txt).

## Confirmed Boost Software License 1.0 material

| Component | turbo_physai path | Direct distribution source and fixed commit | Upstream path | License | NOTICE |
| --- | --- | --- | --- | --- | --- |
| cxx-prettyprint | `kernel/sparse_conv/include/prettyprint.h` | <https://github.com/mit-han-lab/bevfusion> @ `326653dc06e0938edf1aae7d01efcd158ba83de5` | `mmdet3d/ops/spconv/include/prettyprint.h` | Boost-1.0 | No `NOTICE` file at the fixed baseline. |

The complete Boost Software License 1.0 text is in [third_party/licenses/Boost-1.0.txt](third_party/licenses/Boost-1.0.txt). The source file retains Louis Delacroix's original copyright and license notice.

## Provenance policy for future additions

Before public release, every newly introduced third-party or mixed-source file must be resolved to an upstream path and 40-character commit and added here.

All previously identified source files now have a direct upstream path and a fixed 40-character source revision. Any newly introduced third-party or mixed source must be added to this file before public release.

## Hygon modification notice

Where Apache-2.0 source code was modified, the source file retains the upstream copyright and license notice, then adds Hygon copyright and `Modified by Hygon.`. The notice does not change the upstream license.
