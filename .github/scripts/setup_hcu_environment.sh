#!/usr/bin/env bash
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

# Prepare the current ARC job container; no nested Docker or DTK reinstall.
set -eo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
release_tools=false
case "${1:-}" in
  "") ;;
  --release-tools) release_tools=true ;;
  *) echo "Usage: $0 [--release-tools]" >&2; exit 2 ;;
esac
: "${GITHUB_ENV:?Run this setup in GitHub Actions}"
: "${GITHUB_PATH:?Run this setup in GitHub Actions}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_INDEX_URL
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
export PIP_TRUSTED_HOST
DTK_PYPI_INDEX="${DTK_PYPI_INDEX:-https://pypi.sourcefind.cn/nightly/dtk2604}"
export DTK_PYPI_INDEX
DTK_LLVM_URL="${DTK_LLVM_URL:-http://42.228.13.241:18000/ai_cc/Nightly/dtk_llvm-1.0.0%2Bdtk2604.2609201512.gf72f9e-x86_64.run}"
export DTK_LLVM_URL
PYPI_URL="${PYPI_URL:-http://12.12.12.201:9929}"
export PYPI_URL
PYPI_TRUSTED_HOST="${PYPI_TRUSTED_HOST:-12.12.12.201}"
export PYPI_TRUSTED_HOST
DTK_TAG="${DTK_TAG:-2604}"
export DTK_TAG
TORCH_VERSION="${TORCH_VERSION:-2.10.0+das.opt1.dtk2604.2608131652.g28498c}"
export TORCH_VERSION
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0+das185.dtk2604.torch2100.2608130858.gdcd044}"
export TORCHVISION_VERSION
TRITON_VERSION="${TRITON_VERSION:-3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1}"
export TRITON_VERSION
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.4+dtk2604.torch2100.2609231125.g11f180}"
export FLASH_ATTN_VERSION
APEX_VERSION="${APEX_VERSION:-1.7.0+dtk2604.torch2100.2609201739.g9783d4}"
export APEX_VERSION
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.10.0+das.opt1.dtk2604.torch2100.2609021532.gdf6f74}"
export TORCHAUDIO_VERSION
LIGHTOP_WHL_URL="${LIGHTOP_WHL_URL:-https://pypi.sourcefind.cn/das_release/das1.8.5/+f/4a3/1521919e4c764/lightop-0.6.0+das.dtk2604.torch2100.2608121706.g09a4f3-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=4a31521919e4c76405e8dcb5035b55cdc92b2ad6dc5e9f2e72373510835df159}"
export LIGHTOP_WHL_URL
HIPDNN_WHL_URL="${HIPDNN_WHL_URL:-https://hygon-thpcbench-1251001002.cos.ap-shanghai.myqcloud.com/hygon/whl/hipdnn-1.0.1+dtk2604-cp310-cp310-linux_x86_64.whl#sha256=57d0e61cb381bbb244f45140b0dac9775bfc47e33c0eef2acc87627b1b8ea8ec}"
export HIPDNN_WHL_URL

# The base image already supplies DTK and its runtime libraries.
test -f /opt/dtk/env.sh
if [[ ! -e /opt/hyhal && -d /opt/dtk/.hyhal ]]; then
  cp -a /opt/dtk/.hyhal /opt/hyhal
fi
printf '%s\n' /opt/dtk/.hyhal/rocm_smi/lib /opt/dtk/.hyhal/lib \
  > /etc/ld.so.conf.d/hcu-framework.conf
ldconfig
export ROCM_PATH=/opt/dtk DTKROOT=/opt/dtk HIP_PATH=/opt/dtk/hip
export HYHAL_PATH=/opt/hyhal MIOPEN_FIND_MODE=3
export AMDGPU_TARGETS='gfx906;gfx926;gfx928;gfx936;gfx938'
export CMAKE_PREFIX_PATH="/opt/hyhal:${CMAKE_PREFIX_PATH:-}"
export PATH="/opt/dtk/bin:/opt/dtk/llvm/bin:/opt/dtk/hip/bin:/opt/dtk/hip/bin/hipify:/opt/hyhal/bin:/opt/dtk/opencl/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/opt/hyhal/lib:/opt/dtk/opencl/lib:/opt/dtk/dushmem/lib:${LIBRARY_PATH:-}"
export C_INCLUDE_PATH="/opt/dtk/dcc/gcvm/include:/opt/dtk/include:/opt/hyhal/include:/opt/dtk/llvm/include:/opt/dtk/opencl/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="/opt/dtk/dcc/gcvm/include:/opt/dtk/include:/opt/hyhal/include:/opt/dtk/llvm/include:/opt/dtk/dushmem/include:/opt/dtk/opencl/include:${CPLUS_INCLUDE_PATH:-}"
source /opt/dtk/env.sh
curl -fsSL "${DTK_LLVM_URL}" -o /tmp/magi-dtk-llvm.run
# The installer may stop reading yes early; retain its exit status.
(set +o pipefail; yes | bash /tmp/magi-dtk-llvm.run)
rm -f /tmp/magi-dtk-llvm.run
for key in ROCM_PATH DTKROOT HIP_PATH HYHAL_PATH MIOPEN_FIND_MODE \
    AMDGPU_TARGETS CMAKE_PREFIX_PATH LD_LIBRARY_PATH LIBRARY_PATH \
    C_INCLUDE_PATH CPLUS_INCLUDE_PATH; do
  printf '%s=%s\n' "${key}" "${!key}" >> "${GITHUB_ENV}"
done
printf '%s\n' "${PATH//:/$'\n'}" >> "${GITHUB_PATH}"

source /opt/dtk/env.sh
# Preserve docker/ci/Dockerfile's full dependency setup.
# Installing test dependencies does not execute the test suite.
python3 -m pip config set global.index-url "${PIP_INDEX_URL}"
python3 -m pip config set global.trusted-host "${PIP_TRUSTED_HOST}"
python3 -m pip install --upgrade "setuptools==80.9.0" wheel "pip<26"
export LD_LIBRARY_PATH="/opt/dtk/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}"
/usr/bin/python3 -m pip install --no-cache-dir \
  cmake==3.29.2 \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "triton==${TRITON_VERSION}" \
  "flash-attn==${FLASH_ATTN_VERSION}" \
  "apex==${APEX_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${DTK_PYPI_INDEX}"
/usr/bin/python3 -m pip install --no-deps "${LIGHTOP_WHL_URL}" "${HIPDNN_WHL_URL}"
/usr/bin/python3 -m pip install \
  pytest coverage "numpy==1.23.5" "scipy>=1.9.0" PyYAML packaging \
  "Pillow==12.3.0"
# Preserve the dependency installation done before the wheel build in
# run_hcu_ci.sh, including its coverage pin. Do not run that script.
python3 -m pip install -r "${repo_root}/requirements-test.txt"
torch_lib="$(python3 -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent / "lib")')"
echo "LD_LIBRARY_PATH=${torch_lib}:${LD_LIBRARY_PATH:-}" >> "${GITHUB_ENV}"

if "${release_tools}"; then
  # ciupload is hosted on the internal DTK nightly index.
  python3 -m pip install --no-cache-dir \
    --index-url "${PYPI_URL}/nightly/dtk${DTK_TAG}/+simple/" \
    --trusted-host "${PYPI_TRUSTED_HOST}" \
    ciupload
fi

source /opt/dtk/env.sh
python3 - <<'PY'
import os
import sys
from importlib.metadata import version
import torch, torchvision, triton
for name, module in (("torch", torch), ("torchvision", torchvision), ("triton", triton)):
    # The DAS Triton module reports 3.5.0; use wheel metadata.
    actual = version(name) if name == "triton" else module.__version__
    expected = os.environ[f"{name.upper()}_VERSION"]
    assert actual.split("+", 1)[0] == expected.split("+", 1)[0], (name, actual, expected)
    print(f"{name}: {actual}")
print("Triton module version:", triton.__version__)
assert sys.version_info[:2] == (3, 10), sys.version
assert torch.version.hip, "HIP Torch is required"
PY
