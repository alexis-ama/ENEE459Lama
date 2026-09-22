from __future__ import annotations

import json
import re
from typing import Any
import sys

from env import Env, ModuleNotAvailable, getattr_path, read_text, unknown, major_minor

import torch

# The NVIDIA-built PyTorch wheels for Jetson carry a local version segment —
# the part after "+" — that names the NVIDIA container release. A wheel from
# plain PyPI has no such segment. This is a hint, not a proof, which is why the
# probe reports the tag itself alongside the interpretation.
_NV_LOCAL_TAG = re.compile(r"(?:^|\.)nv\d", re.IGNORECASE)

# `# R36 (release), REVISION: 5.0, GCID: ...`
_L4T_RELEASE = re.compile(r"R(\d+)\s*\(release\)", re.IGNORECASE)
_L4T_REVISION = re.compile(r"REVISION:\s*([\d.]+)")

# hepler function
def _split_local_version(raw: str) -> dict[str, Any]:
    if not raw:
        return {"raw": raw, "public": None, "local": None, "nvidia_build": False}
    public, sep, local = raw.partition("+")
    local = local if sep else None
    return {
        "raw": raw,
        "public": public or None,
        "local": local,
        "nvidia_build": bool(local and _NV_LOCAL_TAG.search(local)),
    }

# ---------------------------------------------------------------------------
# The probes.
# ---------------------------------------------------------------------------

def probe_torch(env: Env) -> dict[str, Any]:
    src = "import torch"
    # Step 1
    try:
        env.importer("torch")
    except ModuleNotAvailable:
        unknown(src, f"torch is not importable: {ModuleNotAvailable}")

    # Step 3 (no step 2)
    raw = getattr_path(torch, "__version__")
    if raw:
        version = _split_local_version(str(raw))
    else: 
        version = None

    # Step 4
    cuda_available = getattr_path(torch, "cuda.is_available")
    if callable(cuda_available):
        cuda_available = bool(cuda_available)
    else:
        cuda_available = None

    # Step 5
    device =  None
    cu_ver = None
    if cuda_available:
        cu_name = getattr_path(torch, "cuda.get_device_name")
        cu_ver = getattr_path(torch, "version.cuda")

        if callable(cu_name):
            device = cu_name(0)
        if callable(cu_ver):
            cuda_version = cu_ver(0)

        diagnosis = "torch is installed and sees the GPU"
       
    # Step 7
    if version:
        nv = version["nvidia_build"]
    else:
        nv = False

    if cuda_available:
        diagnosis = "torch is installed and sees the GPU"

    elif not cuda_available:
        diagnosis = """torch is installed but does not expose 
            torch.cuda.is_available"""
    elif nv:
        diagnosis = """this is an NVIDIA build but it cannot see the GPU — 
            the wheel is right, so look at the driver stack, the container, 
            or the user's groups, not at pip"""
    else:
        diagnosis = """this wheel has no NVIDIA local version tag and cannot 
            see the GPU — it is almost certainly a stock PyPI wheel and must 
            be replaced from the Jetson index"""

    # Step 6
    ret = {
        "value": raw,
        "source": src,
        "status": "ok",
        "version": version,
        "cuda_available": cuda_available,
        "cuda_version": cu_ver,
        "device_name": device,
        "diagnosis": diagnosis
    }

    if not raw:
        ret["detail"] = "torch imported but exposes no __version__"

    return ret

def probe_cuda(env: Env) -> dict[str, Any]:
    src = "/usr/local/cuda/version.json"

    raw = read_text(env.root, src)
    if not raw:
        unknown(src, "CUDA toolkit manifest absent — no toolkit installed at /usr/local/cuda")
    
    try:
        if raw: # need to include for some reason
            data = json.loads(raw)
    except (ValueError, json.JSONDecodeError):
        unknown(src, "CUDA toolkit manifest is present but not valid JSON")
    # pass
    


def probe_opencv(env: Env) -> dict[str, Any]:
    #write your code here
    pass


def probe_tensorrt(env: Env) -> dict[str, Any]:
    # write your code here
    pass


def probe_l4t(env: Env) -> dict[str, Any]:
    # write your code here
    pass

## for debugging - uncomment the following lines for debugging.
if __name__ == "__main__":
    env = Env.real()
    out = probe_torch(env)
    print(out)

# for generating system_report.json
# if __name__ == "__main__":
#     # calling base environment
#     env = Env.real()

#     # testing probes
#     report = {
#         "probe_torch": probe_torch(env),
#         "probe_cuda": probe_cuda(env),
#         "probe_opencv": probe_opencv(env),
#         "probe_tensorrt": probe_tensorrt(env),
#         "probe_l4t": probe_l4t(env),
#     }
    
    # path = "system_report.json"
    # with open(path, "w", encoding="utf-8") as f:
    #     json.dump(report, f, indent=4)

