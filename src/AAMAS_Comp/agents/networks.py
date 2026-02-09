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
    ortho_init: bool = True,
    output_gain: float = 1.0,
) -> nn.Sequential:
    """Build a simple MLP with optional orthogonal initialization (SB3-style).

    Args:
        in_features: Dimension of the input.
        out_features: Dimension of the output.
        hidden_sizes: Widths of hidden layers.
        activation: Name of the activation function (from ``torch.nn``).
        output_activation: Optional activation after the final layer.
        device: Device to place the module on.
        dtype: Optional dtype for network parameters.
        ortho_init: Use orthogonal initialization (SB3 default). Hidden layers
            get gain=sqrt(2), output layer gets ``output_gain``.
        output_gain: Gain for the output layer when ``ortho_init=True``.
            Use 0.01 for actor (near-uniform initial policy) and 1.0 for critic.

    Returns:
        A ``nn.Sequential`` MLP.
    """
    act_cls = getattr(nn, activation)
    layers: List[nn.Module] = []

    prev = in_features
    for h in hidden_sizes:
        linear = nn.Linear(prev, h, device=device)
        if ortho_init:
            nn.init.orthogonal_(linear.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(linear.bias, 0.0)
        layers.append(linear)
        layers.append(act_cls())
        prev = h

    output_linear = nn.Linear(prev, out_features, device=device)
    if ortho_init:
        nn.init.orthogonal_(output_linear.weight, gain=output_gain)
        nn.init.constant_(output_linear.bias, 0.0)
    layers.append(output_linear)

    if output_activation is not None:
        layers.append(getattr(nn, output_activation)())

    net = nn.Sequential(*layers)
    
    # Convert to target dtype after construction if specified
    if dtype is not None:
        net = net.to(dtype=dtype)
    
    return net
