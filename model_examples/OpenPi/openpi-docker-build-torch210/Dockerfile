# syntax=docker/dockerfile:1
# Build context:
#   .
#   ├── Dockerfile
#   └── requirements.txt
#
# `/data` (dataset/checkpoints) and `/opt/hyhal` (host driver runtime) are
# deliberately runtime mounts.  Neither is copied into this image.
FROM harbor.sourcefind.cn:5443/hcu/admin/base/custom:dit-vla-wm-base-ubuntu22.04-dtk26.04-torch2.10-py3.10

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG AICC_URL=http://42.228.13.241:18000/ai_cc/nightly/dtk_llvm_630f2f_202608141530.run
ARG DTK_BLAS_URL=http://42.228.13.241:18000/dtk-archived/nightly/dtk-26.04/0921/DTK-26.04-blas-nightly-0921-cut-ubuntu20.04-x86_64.tar.gz

# Keep the exact reference revision.  Override LEROBOT_GIT_URL for a customer
# Git mirror if github.com is inaccessible; hf-mirror only mirrors HF assets,
# not GitHub repositories.
ARG LEROBOT_GIT_URL=https://github.com/huggingface/lerobot.git
ARG LEROBOT_REV=0cf864870cf29f4738d3ade893e6fd13fbd7cdb5

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PATH=/opt/dtk/bin:/opt/dtk/aillvm/bin:${PATH} \
    OPENPI_DTK_BLAS_LIB_DIR=/opt/openpi/dtk-26.04-blas/lib

# Keep the Torch, TorchVision, FlashAttention and Triton stack from the base image.

RUN python -m pip install --no-cache-dir 'Jinja2>=3.1,<4'

# Triton 3.6 emits kernels through aicc.  The installer places it under the
# existing /opt/dtk tree; no /opt/hyhal host driver files are baked into image.
RUN set -eux; \
    curl -fsSL --retry 3 --retry-delay 2 "$AICC_URL" -o /tmp/aicc.run; \
    chmod 0755 /tmp/aicc.run; \
    yes | /tmp/aicc.run; \
    rm -f /tmp/aicc.run; \
    test -x /opt/dtk/bin/aicc

COPY requirements.txt /tmp/requirements.txt

# Install all non-binary OpenPI runtime dependencies after AICC.
RUN python -m pip install --no-cache-dir --index-url "$PIP_INDEX_URL" \
        -r /tmp/requirements.txt \
    && python -m pip install --no-cache-dir --index-url "$PIP_INDEX_URL" \
        transformers==4.53.2 tokenizers==0.21.4 \
    && python -m pip install --no-cache-dir --index-url "$PIP_INDEX_URL" --no-deps gym-aloha==0.1.1

# Install LeRobot only after the vendor torch stack is already present, so pip
# treats it as satisfying LeRobot's torch/torchvision requirements.
RUN python -m pip install --no-cache-dir \
        --no-deps "lerobot @ git+${LEROBOT_GIT_URL}@${LEROBOT_REV}"

# Install only the standalone BLAS/BLASLT package after the Python runtime
# stack.  The base image already contains the complete DTK runtime.
RUN set -eux; \
    mkdir -p /opt/openpi/dtk-26.04-blas-raw; \
    curl -fsSL --retry 3 --retry-delay 2 "$DTK_BLAS_URL" -o /tmp/dtk-blas.tar.gz; \
    tar -xzf /tmp/dtk-blas.tar.gz -C /opt/openpi/dtk-26.04-blas-raw; \
    rm -f /tmp/dtk-blas.tar.gz; \
    BLAS_LIB_DIR="$(find /opt/openpi/dtk-26.04-blas-raw -type f -name 'librocblas.so*' -printf '%h\n' | head -n 1)"; \
    test -n "$BLAS_LIB_DIR"; \
    mkdir -p /opt/openpi/dtk-26.04-blas; \
    ln -s "$BLAS_LIB_DIR" /opt/openpi/dtk-26.04-blas/lib; \
    test -e /opt/openpi/dtk-26.04-blas/lib/librocblas.so


CMD ["bash"]
