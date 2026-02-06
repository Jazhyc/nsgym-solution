"""Reusable network building blocks for RL agents.

This module provides configurable actor and critic network factories
so that the architecture can be easily swapped out via Hydra configs.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn


def make_mlp(
    in_features: int,
    out_features: int,
    hidden_sizes: Sequence[int] = (256, 256),
    activation: str = "Tanh",
    output_activation: str | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> nn.Sequential:
    """Build a simple MLP.

    Args:
        in_features: Dimension of the input.
        out_features: Dimension of the output.
        hidden_sizes: Widths of hidden layers.
        activation: Name of the activation function (from ``torch.nn``).
        output_activation: Optional activation after the final layer.
        device: Device to place the module on.
        dtype: Optional dtype for network parameters.

    Returns:
        A ``nn.Sequential`` MLP.
    """
    act_cls = getattr(nn, activation)
    layers: List[nn.Module] = []

    prev = in_features
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h, device=device))
        layers.append(act_cls())
        prev = h

    layers.append(nn.Linear(prev, out_features, device=device))

    if output_activation is not None:
        layers.append(getattr(nn, output_activation)())

    net = nn.Sequential(*layers)
    
    # Convert to target dtype after construction if specified
    if dtype is not None:
        net = net.to(dtype=dtype)
    
    return net
