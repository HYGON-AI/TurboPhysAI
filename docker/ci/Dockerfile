# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

# TurboPhysAI CI: supplied HCU framework plus build and test dependencies.
# Build from the repository root:
# docker build --network=host -f docker/ci/Dockerfile \
#   -t turbophysai-ci:hcu-ci-local .
# Build from the repository root so the CI verification script can be copied.
ARG BASE_IMAGE=10.16.1.152:5000/hcu/base/dtk:26.04-ubuntu22.04-py3.10

FROM ${BASE_IMAGE} AS framework
SHELL ["/bin/bash", "-c"]
ARG TARGETS_EXTRA
ARG TARGETS_OVERRIDE
ARG ARTIFACT_HOST=42.228.13.241

ENV ROCM_PATH=/opt/dtk
RUN test -f /opt/dtk/env.sh

ENV HYHAL_PATH=/opt/hyhal \
    DTKROOT=${ROCM_PATH} \
    AMDGPU_TARGETS=${TARGETS_OVERRIDE:-"gfx906;gfx926;gfx928;gfx936;gfx938${TARGETS_EXTRA}"} \
    HIP_PATH=${ROCM_PATH}/hip \
    MIOPEN_FIND_MODE=3
ENV CMAKE_PREFIX_PATH=/opt/hyhal:$CMAKE_PREFIX_PATH
ENV PATH=${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${ROCM_PATH}/hip/bin:${ROCM_PATH}/hip/bin/hipify:${HYHAL_PATH}/bin:${ROCM_PATH}/opencl/bin:$PATH
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${HYHAL_PATH}/lib:${HYHAL_PATH}/lib64:${ROCM_PATH}/dushmem/lib:${ROCM_PATH}/opencl/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${ROCM_PATH}/dcc/gcvm/lib:${ROCM_PATH}/hip/lib:${ROCM_PATH}/llvm/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=${HYHAL_PATH}/lib:${ROCM_PATH}/opencl/lib:${ROCM_PATH}/dushmem/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}
ENV C_INCLUDE_PATH=${ROCM_PATH}/include:${HYHAL_PATH}/include:${ROCM_PATH}/llvm/include:${ROCM_PATH}/opencl/include${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}
ENV C_INCLUDE_PATH=${ROCM_PATH}/dcc/gcvm/include:${C_INCLUDE_PATH}
ENV CPLUS_INCLUDE_PATH=${ROCM_PATH}/dcc/gcvm/include:${ROCM_PATH}/include:${HYHAL_PATH}/include:${ROCM_PATH}/llvm/include:${ROCM_PATH}/dushmem/include:${ROCM_PATH}/opencl/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}

# Match the runtime library setup used by the autonomous-driving companion image.
RUN if [[ ! -e /opt/hyhal && -d /opt/dtk/.hyhal ]]; then \
        cp -a /opt/dtk/.hyhal /opt/hyhal; \
    fi && \
    printf '%s\n' /opt/dtk/.hyhal/rocm_smi/lib /opt/dtk/.hyhal/lib \
      > /etc/ld.so.conf.d/hcu-framework.conf && ldconfig

RUN source /opt/dtk/env.sh && \
    wget -q -O /tmp/magi-dtk-llvm.run \
      "http://${ARTIFACT_HOST}:18000/ai_cc/Nightly/dtk_llvm-1.0.0%2Bdtk2604.2609201512.gf72f9e-x86_64.run" && \
    yes | bash /tmp/magi-dtk-llvm.run && \
    rm -f /tmp/magi-dtk-llvm.run

RUN source /opt/dtk/env.sh && \
    export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-} && \
    /usr/bin/python3 -m pip install --no-cache-dir 'pip<26' setuptools wheel && \
    /usr/bin/python3 -m pip install --no-cache-dir \
      cmake==3.29.2 \
      torch==2.10.0+das.opt1.dtk2604.2608131652.g28498c \
      torchvision==0.25.0+das185.dtk2604.torch2100.2608130858.gdcd044 \
      triton==3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1 \
      flash-attn==2.8.4+dtk2604.torch2100.2609231125.g11f180 \
      apex==1.7.0+dtk2604.torch2100.2609201739.g9783d4 \
      torchaudio==2.10.0+das.opt1.dtk2604.torch2100.2609021532.gdf6f74 \
      --index-url https://pypi.sourcefind.cn/nightly/dtk2604

ARG LIGHTOP_WHL_URL=https://pypi.sourcefind.cn/das_release/das1.8.5/+f/4a3/1521919e4c764/lightop-0.6.0+das.dtk2604.torch2100.2608121706.g09a4f3-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=4a31521919e4c76405e8dcb5035b55cdc92b2ad6dc5e9f2e72373510835df159
RUN /usr/bin/python3 -m pip install --no-cache-dir --no-deps "${LIGHTOP_WHL_URL}"

ARG HIPDNN_WHL_URL=https://hygon-thpcbench-1251001002.cos.ap-shanghai.myqcloud.com/hygon/whl/hipdnn-1.0.1+dtk2604-cp310-cp310-linux_x86_64.whl#sha256=57d0e61cb381bbb244f45140b0dac9775bfc47e33c0eef2acc87627b1b8ea8ec
RUN /usr/bin/python3 -m pip install --no-cache-dir --no-deps "${HIPDNN_WHL_URL}"

RUN printf '%s\n' 'source /opt/dtk/env.sh' >> /root/.bashrc && \
    printf 'dtk    : %s\n' "$(cat /opt/dtk/.info/rocm_version)" > /etc/image.info

# Import checks run without requiring an HCU device during image construction.
RUN source /opt/dtk/env.sh && \
    /usr/bin/python3 -c 'import sys, torch, torchvision, torchaudio, triton, hipdnn; assert sys.version_info[:2] == (3, 10), sys.version; assert torch.version.hip, "HIP Torch is required"; print("torch:", torch.__version__, "hip:", torch.version.hip)'

WORKDIR /workspace
CMD ["/bin/bash"]


# Add TurboPhysAI build/test tools to the supplied Torch 2.10 framework.
FROM framework AS ci
ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ENV LANG=C.UTF-8 \
    PYTHONNOUSERSITE=1 \
    TORCH_CUDA_ARCH_LIST=9.0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl git libnuma-dev ninja-build \
      python3-dev python3-tk && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m pip config set global.index-url "${PIP_INDEX_URL}" && \
    python3 -m pip config set global.trusted-host "${PIP_TRUSTED_HOST}" && \
    python3 -m pip install --upgrade "setuptools==80.9.0" wheel 'pip<26' && \
    python3 -m pip install \
      pytest coverage numpy==1.23.5 "scipy>=1.9.0" PyYAML packaging \
      "Pillow==12.3.0"
COPY docker/ci/verify-ci-environment.py /usr/local/bin/verify-turbophysai-ci
RUN chmod 0755 /usr/local/bin/verify-turbophysai-ci && \
    source /opt/dtk/env.sh && \
    /usr/bin/python3 -c "import numpy as np, torch; assert torch.__version__.split('+', 1)[0] == '2.10.0'; torch.from_numpy(np.zeros(1))" && \
    /usr/bin/python3 -c "import importlib.util; assert importlib.util.find_spec('hipdnn'); assert importlib.util.find_spec('lightop')"
WORKDIR /workspace/TurboPhysAI
CMD ["bash"]
