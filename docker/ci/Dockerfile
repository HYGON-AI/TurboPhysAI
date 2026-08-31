# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

# Minimal TurboPhysAI CI environment. It intentionally does not contain the
# TurboPhysAI source or wheel: every CI job installs and tests its checked-out
# candidate revision.
ARG L0_IMAGE
FROM ${L0_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ARG COS_WHL_BASE=https://hygon-thpcbench-1251001002.cos.ap-shanghai.myqcloud.com/hygon/whl
ARG DTK_PYPI_INDEX=https://pypi.sourcefind.cn/release/dtk2604/+simple/
ARG TORCH_WHL_URL=${COS_WHL_BASE}/torch-2.7.1+das.opt1.dtk2604.2605091817.g39c19f-cp310-cp310-manylinux_2_28_x86_64.whl
ARG TRITON_WHL_URL=${COS_WHL_BASE}/triton-3.2.0+das.opt1.dtk2604.torch271.2605122039.gc13c4e-cp310-cp310-manylinux_2_28_x86_64.whl
ARG TORCHVISION_VERSION=0.22.0+das.opt1.dtk2604.torch271
ARG HIPDNN_WHL_URL=${COS_WHL_BASE}/hipdnn-1.0.1+dtk2604-cp310-cp310-linux_x86_64.whl
ARG LIGHTOP_WHL_URL=${COS_WHL_BASE}/lightop-0.6.0%2Bdas.dtk2604-cp310-cp310-linux_x86_64.whl

ENV LANG=C.UTF-8 \
    PYTHONNOUSERSITE=1 \
    TORCH_CUDA_ARCH_LIST=9.0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10 \
    LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/opt/dtk/cuda/cuda-12/lib64:${LD_LIBRARY_PATH}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl git libnuma-dev ninja-build \
      python3-dev python3-tk && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip config set global.index-url "${PIP_INDEX_URL}" && \
    python3 -m pip config set global.trusted-host "${PIP_TRUSTED_HOST}" && \
    python3 -m pip install --upgrade "setuptools==80.9.0" wheel pip && \
    /usr/bin/python3 -m pip install \
      "${TORCH_WHL_URL}" \
      "${TRITON_WHL_URL}" \
      "${HIPDNN_WHL_URL}" && \
    /usr/bin/python3 -m pip install --no-deps \
      --index-url "${DTK_PYPI_INDEX}" \
      "torchvision==${TORCHVISION_VERSION}" && \
    /usr/bin/python3 -m pip install --ignore-installed --no-deps \
      "${LIGHTOP_WHL_URL}" && \
    /usr/bin/python3 -m pip install \
      pytest coverage numpy==1.23.5 "scipy>=1.9.0" PyYAML packaging \
      "Pillow==12.3.0"

# Needed while setup.py imports torch during the CI job's wheel build.
RUN test -d /opt/dtk/.hyhal/rocm_smi/lib && \
    printf '%s\n' '/opt/dtk/.hyhal/rocm_smi/lib' \
      > /etc/ld.so.conf.d/turbophysai-dtk.conf && \
    ldconfig

COPY docker/ci/verify-ci-environment.py /usr/local/bin/verify-turbophysai-ci
RUN chmod 0755 /usr/local/bin/verify-turbophysai-ci && \
    /usr/bin/python3 -c "import numpy as np, torch; assert torch.__version__.startswith('2.7.1'); torch.from_numpy(np.zeros(1))" && \
    /usr/bin/python3 -c "import importlib.util; assert importlib.util.find_spec('hipdnn'); assert importlib.util.find_spec('lightop')"

WORKDIR /workspace/TurboPhysAI
CMD ["bash"]
