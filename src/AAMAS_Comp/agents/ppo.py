"""TorchRL-based PPO agent.

Provides:
- ``make_ppo_models``: factory that builds the actor (policy), critic (value),
  and the ``ClipPPOLoss`` module from a flat ``omegaconf.DictConfig``.
- ``PPOAgent``: thin wrapper that conforms to the competition's
  ``ModelFreeAgent`` interface so a trained policy can be used at evaluation
  time.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.envs import EnvBase
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from AAMAS_Comp.agents.networks import make_mlp
from AAMAS_Comp.base_agent import ModelFreeAgent


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_actor(
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    activation: str,
    device: torch.device | str,
    action_spec: Any | None = None,
) -> ProbabilisticActor:
    """Build a stochastic Gaussian actor wrapped as a ``ProbabilisticActor``.

    The underlying network outputs ``(loc, scale)`` which are consumed by
    a ``TanhNormal`` distribution that respects action-space bounds.
    """
    net = nn.Sequential(
        make_mlp(
            in_features=obs_dim,
            out_features=2 * act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            device=device,
        ),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(
        net,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )

    dist_kwargs = {}
    if action_spec is not None:
        dist_kwargs["low"] = action_spec.space.low
        dist_kwargs["high"] = action_spec.space.high

    actor = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs=dist_kwargs,
        return_log_prob=True,
    )
    return actor


def make_critic(
    obs_dim: int,
    hidden_sizes: Sequence[int],
    activation: str,
    device: torch.device | str,
) -> ValueOperator:
    """Build a state-value critic ``V(s)``."""
    net = make_mlp(
        in_features=obs_dim,
        out_features=1,
        hidden_sizes=hidden_sizes,
        activation=activation,
        device=device,
    )
    return ValueOperator(module=net, in_keys=["observation"])


def make_ppo_models(
    env: EnvBase,
    cfg: DictConfig,
    device: torch.device | str = "cpu",
    network_device: torch.device | str | None = None,
) -> dict:
    """Construct actor, critic, advantage estimator, loss, and optimizer.

    Args:
        env: A TorchRL environment (used for specs).
        cfg: Flat Hydra config that must contain the keys listed in
            ``config/``.
        device: Device for environment/collector (used for data transfers).
        network_device: Device for neural networks. Defaults to ``device``
            if not specified, or uses ``cfg.agent.network_device`` if available.

    Returns:
        Dictionary with keys ``actor``, ``critic``, ``advantage``,
        ``loss_module``, ``optimizer``, ``scheduler``.
    """
    if network_device is None:
        # Try to get from config, otherwise default to device
        if hasattr(cfg, "agent") and hasattr(cfg.agent, "network_device"):
            network_device = cfg.agent.network_device
        else:
            network_device = device
    
    network_device = torch.device(network_device) if isinstance(network_device, str) else network_device
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    hidden_sizes = list(cfg.agent.hidden_sizes)
    activation = cfg.agent.activation

    actor = make_actor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        device=network_device,
        action_spec=env.action_spec_unbatched,
    )

    critic = make_critic(
        obs_dim=obs_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        device=network_device,
    )

    # -- Advantage estimation (GAE) ------------------------------------------
    advantage = GAE(
        gamma=cfg.agent.gamma,
        lmbda=cfg.agent.gae_lambda,
        value_network=critic,
        average_gae=True,
    )

    # -- PPO clipped loss -----------------------------------------------------
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.agent.clip_epsilon,
        entropy_bonus=cfg.agent.entropy_coeff > 0,
        entropy_coeff=cfg.agent.entropy_coeff,
        critic_coeff=cfg.agent.critic_coeff,
        loss_critic_type=cfg.agent.loss_critic_type,
    )

    # -- Optimizer & LR scheduler ---------------------------------------------
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.agent.lr)

    total_iters = cfg.collector.total_frames // cfg.collector.frames_per_batch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_iters,
        eta_min=cfg.agent.lr_min,
    )

    return {
        "actor": actor,
        "critic": critic,
        "advantage": advantage,
        "loss_module": loss_module,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


# ---------------------------------------------------------------------------
# Competition-compatible wrapper
# ---------------------------------------------------------------------------

class PPOAgent(ModelFreeAgent):
    """Wraps a trained TorchRL actor for the AAMAS competition interface.

    Args:
        actor: A ``ProbabilisticActor`` (or any ``TensorDictModule`` that maps
            ``observation -> action``).
        device: Torch device the actor lives on.
        deterministic: If ``True`` use the mean action (no sampling).
    """

    def __init__(
        self,
        actor: ProbabilisticActor,
        device: torch.device | str = "cpu",
        deterministic: bool = True,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.device = torch.device(device)
        self.deterministic = deterministic

    def get_action(self, obs: Dict) -> np.ndarray:
        state = obs["state"]
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        from tensordict import TensorDict
        td = TensorDict({"observation": state_t}, batch_size=[])

        from torchrl.envs.utils import ExplorationType, set_exploration_type

        explore = ExplorationType.DETERMINISTIC if self.deterministic else ExplorationType.RANDOM
        with set_exploration_type(explore), torch.no_grad():
            td = self.actor(td)

        action = td["action"].cpu().numpy()
        return action

    # ---- Persistence helpers ------------------------------------------------

    @classmethod
    def load(cls, path: str, device: str = "cpu", deterministic: bool = True) -> "PPOAgent":
        """Load a saved actor checkpoint.

        Args:
            path: Path to a ``.pt`` file saved via ``PPOAgent.save()``.
            device: Device to map the model onto.
            deterministic: Use deterministic actions.

        Returns:
            A ready-to-use ``PPOAgent``.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        actor = checkpoint["actor"]
        actor.to(device)
        return cls(actor=actor, device=device, deterministic=deterministic)

    def save(self, path: str) -> None:
        """Persist the actor so it can be loaded later."""
        torch.save({"actor": self.actor}, path)

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
