#!/usr/bin/env bash
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact_dir="${TURBOPHYSAI_CI_ARTIFACT_DIR:-${repo_root}/ci-artifacts}"
verify_dir="$(mktemp -d /tmp/turbophysai-verify.XXXXXX)"
test_dir="$(mktemp -d /tmp/turbophysai-test.XXXXXX)"
build_root="$(mktemp -d /tmp/turbophysai-build.XXXXXX)"

mkdir -p "${artifact_dir}/dist"
log_file="${artifact_dir}/turbophysai-ci.log"
exec > >(tee "${log_file}") 2>&1

cleanup() {
  status=$?
  rm -rf -- "${verify_dir}" "${test_dir}" "${build_root}"
  if [[ "${TURBOPHYSAI_HOST_UID:-}" =~ ^[0-9]+$ && \
        "${TURBOPHYSAI_HOST_GID:-}" =~ ^[0-9]+$ ]]; then
    chown -R "${TURBOPHYSAI_HOST_UID}:${TURBOPHYSAI_HOST_GID}" "${artifact_dir}" 2>/dev/null || true
  fi
  chmod -R a+rX "${artifact_dir}" 2>/dev/null || true
  trap - EXIT
  exit "${status}"
}
trap cleanup EXIT

echo "===== Source identity ====="
echo "repository=${repo_root}"
echo "workflow_sha=${GITHUB_SHA:-unset}"
echo "workflow_ref_name=${GITHUB_REF_NAME:-unset}"
echo "workflow_ref_type=${GITHUB_REF_TYPE:-unset}"
echo "container_image=${TURBOPHYSAI_CI_IMAGE:-unset}"
git -c safe.directory="${repo_root}" -C "${repo_root}" rev-parse HEAD
git config --global --add safe.directory "${repo_root}"

echo "===== DTK and build-tool environment ====="
test -f /opt/dtk/env.sh
# shellcheck disable=SC1091
source /opt/dtk/env.sh
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export MAX_JOBS="${MAX_JOBS:-4}"
export ROCM_HOME="${ROCM_HOME:-${ROCM_PATH:-/opt/dtk}}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH

python3 --version
python3 -m pip --version
hipcc --version
cmake --version
ninja --version
gcc --version | head -n 1
git --version

if command -v hy-smi >/dev/null 2>&1; then
  hy-smi
elif command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi
else
  echo "No hy-smi or rocm-smi command is available; torch will report the HCU model."
fi

echo "===== Verify the CI image does not preinstall TurboPhysAI ====="
cd "${verify_dir}"
env -u PYTHONPATH python3 - <<'PY'
import importlib.util

spec = importlib.util.find_spec("turbo_physai")
assert spec is None, f"the CI image must not preinstall turbo_physai: {spec}"
PY

echo "===== Install declared build and test dependencies ====="
python3 -m pip install -r "${repo_root}/requirements-dev.txt"
python3 - <<'PY'
import setuptools
import wheel

print("setuptools:", setuptools.__version__)
print("wheel:", wheel.__version__)
PY

echo "===== Build wheel from the tested checkout ====="
# Build in a disposable exact copy. Native-extension builds may create HIPify
# outputs and egg-info files; keeping them out of the mounted checkout avoids
# cross-run contamination on a persistent self-hosted runner.
cp -a "${repo_root}/." "${build_root}/"
git config --global --add safe.directory "${build_root}"
python3 -m pip wheel \
  --no-build-isolation \
  --no-deps \
  --wheel-dir "${artifact_dir}/dist" \
  "${build_root}" 2>&1 | tee "${artifact_dir}/build.log"

mapfile -t wheels < <(find "${artifact_dir}/dist" -maxdepth 1 -type f -name 'turbo_physai-*.whl' -print)
if [[ "${#wheels[@]}" -ne 1 ]]; then
  echo "Expected exactly one TurboPhysAI wheel, found ${#wheels[@]}" >&2
  printf 'wheel=%s\n' "${wheels[@]:-}"
  exit 1
fi

python3 -m pip install --force-reinstall --no-deps "${wheels[0]}"

echo "===== Verify imports outside the checkout ====="
cd "${verify_dir}"
env -u PYTHONPATH TURBOPHYSAI_REPO_ROOT="${repo_root}" python3 - <<'PY'
import os
from pathlib import Path

import turbo_physai
import turbo_physai.ops

repo_root = Path(os.environ["TURBOPHYSAI_REPO_ROOT"]).resolve()
for module in (turbo_physai, turbo_physai.ops):
    module_path = Path(module.__file__).resolve()
    print(f"{module.__name__}: {module_path}")
    assert not module_path.is_relative_to(repo_root), module_path
PY

echo "===== Verify required versions and single-HCU visibility ====="
env -u PYTHONPATH TURBOPHYSAI_REPO_ROOT="${repo_root}" python3 - <<'PY'
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.version import Version
import hipdnn
import lightop
import torch
import torchvision
import triton
import turbo_physai
import turbo_physai.ops


def require_release(actual, expected, component):
    parsed = Version(actual.split("+", 1)[0])
    assert parsed.release == expected, f"{component}: expected {expected}, got {actual}"


require_release(torch.__version__, (2, 7, 1), "PyTorch")
require_release(torchvision.__version__, (0, 22, 0), "torchvision")
require_release(triton.__version__, (3, 2, 0), "Triton")
assert torch.version.hip is not None, "DAS/HCU PyTorch with HIP support is required"
assert torch.cuda.is_available(), "HCU is not available through torch.cuda"
assert torch.cuda.device_count() >= 1, "at least one visible HCU is required"

print("Python package versions:")
for distribution in (
    "torch",
    "torchvision",
    "triton",
    "hipdnn",
    "lightop",
    "numpy",
    "scipy",
    "PyYAML",
    "packaging",
    "Pillow",
    "pytest",
    "coverage",
):
    try:
        print(f"  {distribution}={version(distribution)}")
    except PackageNotFoundError:
        print(f"  {distribution}=installed without distribution metadata")

print("torch HIP runtime:", torch.version.hip)
print("HCU count:", torch.cuda.device_count())
print("HCU model:", torch.cuda.get_device_name(0))
print("hipdnn:", hipdnn.__file__)
print("lightop:", lightop.__file__)
print("TurboPhysAI:", turbo_physai.__file__)
print("TurboPhysAI ops:", turbo_physai.ops.__file__)

repo_root = Path(os.environ["TURBOPHYSAI_REPO_ROOT"]).resolve()
for module in (turbo_physai, turbo_physai.ops):
    assert not Path(module.__file__).resolve().is_relative_to(repo_root)
PY

echo "===== Run the complete non-HCU and single-HCU test suite ====="
cd "${test_dir}"
env -u PYTHONPATH python3 -m pytest \
  --import-mode=importlib \
  -p no:cacheprovider \
  "${repo_root}/test" \
  -vv \
  --junitxml="${artifact_dir}/pytest-junit.xml" 2>&1 | tee "${artifact_dir}/pytest.log"

echo "===== CI completed successfully ====="
