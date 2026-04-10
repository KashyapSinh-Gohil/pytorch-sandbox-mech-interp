# Copyright (c) 2026, Kashyapsinh Gohil
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Mech Interp Environment.

This package exposes the PyTorchSandbox mechanistic interpretability benchmark
as an OpenEnv-compatible environment.
"""

__version__ = "0.1.0"

from .models import MechInterpAction, MechInterpObservation, InterpState

try:
    from .client import MechInterpEnv
except ImportError:
    MechInterpEnv = None

try:
    from .server.mech_interp_environment import MechInterpEnvironment
except ImportError:
    try:
        from server.mech_interp_environment import MechInterpEnvironment
    except ImportError:
        MechInterpEnvironment = None

__all__ = [
    "MechInterpAction",
    "MechInterpObservation",
    "InterpState",
    "MechInterpEnv",
    "MechInterpEnvironment",
]
