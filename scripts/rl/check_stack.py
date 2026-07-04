"""Boot Isaac Sim headlessly and verify the installed RL training stack."""

from __future__ import annotations

import importlib.metadata
import json
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher


app = AppLauncher(headless=True).app
try:
    import isaaclab  # noqa: F401
    import isaaclab_rl  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    import rsl_rl  # noqa: F401
    import torch

    report = {
        "isaaclab": importlib.metadata.version("isaaclab"),
        "isaaclab_rl": importlib.metadata.version("isaaclab_rl"),
        "isaaclab_tasks": importlib.metadata.version("isaaclab_tasks"),
        "rsl_rl": importlib.metadata.version("rsl-rl-lib"),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print("XRC_RL_STACK " + json.dumps(report, sort_keys=True), flush=True)
finally:
    app.close()
