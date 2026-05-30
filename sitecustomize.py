"""Compatibility shims loaded automatically by Python when running from the repo root.

This keeps older third-party dependencies working on newer torchvision releases
without touching site-packages.
"""
from __future__ import annotations

import sys
import types


def _install_torchvision_functional_tensor_shim() -> None:
    try:
        from torchvision.transforms import functional as tv_functional
    except Exception:
        return

    module_name = 'torchvision.transforms.functional_tensor'
    if module_name in sys.modules:
        return

    shim = types.ModuleType(module_name)
    shim.__dict__['__all__'] = []

    # Mirror the modern torchvision.transforms.functional API so legacy imports
    # like `from torchvision.transforms.functional_tensor import rgb_to_grayscale`
    # keep working.
    for name in dir(tv_functional):
        if name.startswith('_'):
            continue
        try:
            setattr(shim, name, getattr(tv_functional, name))
            shim.__all__.append(name)
        except Exception:
            continue

    sys.modules[module_name] = shim


_install_torchvision_functional_tensor_shim()
