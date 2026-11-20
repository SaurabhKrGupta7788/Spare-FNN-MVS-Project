"""Print the versions and device availability needed by the project."""

from __future__ import annotations

import importlib

MODULES = ["torch", "transformers", "datasets", "huggingface_hub", "numpy", "pandas", "matplotlib", "scipy"]

for module_name in MODULES:
    try:
        module = importlib.import_module(module_name)
        print(f"{module_name:18s} {getattr(module, '__version__', 'installed')}")
    except Exception as exc:  # report missing/broken imports clearly
        print(f"{module_name:18s} MISSING ({type(exc).__name__}: {exc})")

try:
    import torch

    print(f"CUDA available:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device:        {torch.cuda.get_device_name(0)}")
except Exception:
    pass
