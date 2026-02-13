"""TorchRL-based PPO agent.

Provides:
- ``make_ppo_models``: factory that builds the actor (policy), critic (value),
  and the ``ClipPPOLoss`` module from a flat ``omegaconf.DictConfig``.
- ``PPOAgent``: thin wrapper that conforms to the competition's
  ``ModelFreeAgent`` interface so a trained policy can be used at evaluation
  time.
"""

from __future__ import annotations

import logging
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

log = logging.getLogger(__name__)


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
    dtype: torch.dtype | None = None,
) -> ProbabilisticActor:
    """Build a stochastic Gaussian actor wrapped as a ``ProbabilisticActor``.

    The underlying network outputs ``(loc, scale)`` which are consumed by
    a ``TanhNormal`` distribution that respects action-space bounds.
    """
    # Create MLP and convert to dtype BEFORE wrapping in Sequential with NormalParamExtractor
    mlp = make_mlp(
        in_features=obs_dim,
        out_features=2 * act_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        device=device,
        dtype=dtype,
        ortho_init=True,
        output_gain=0.01,  # Small gain → near-uniform initial policy (SB3 default)
    )
    
    # Wrap with NormalParamExtractor after dtype conversion
    net = nn.Sequential(mlp, NormalParamExtractor())

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
    dtype: torch.dtype | None = None,
) -> ValueOperator:
    """Build a state-value critic ``V(s)``."""
    net = make_mlp(
        in_features=obs_dim,
        out_features=1,
        hidden_sizes=hidden_sizes,
        activation=activation,
        device=device,
        dtype=dtype,
        ortho_init=True,
        output_gain=1.0,  # Critic output gain (SB3 default)
    )
    critic = ValueOperator(module=net, in_keys=["observation"])
    
    return critic


def make_shared_actor_critic(
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    activation: str,
    device: torch.device | str,
    action_spec: Any | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[ProbabilisticActor, ValueOperator]:
    """Build actor and critic that share hidden-layer weights.

    The trunk (all hidden layers) is the *same* ``nn.Module`` object
    embedded in both networks, so their parameters are shared.  Each
    network reads ``"observation"`` and can be used directly with
    TorchRL's GAE / ClipPPOLoss — no manual trunk forwarding required.
    """
    # ── Shared trunk (all hidden layers) ──────────────────────────────
    trunk_net = make_mlp(
        in_features=obs_dim,
        out_features=hidden_sizes[-1],
        hidden_sizes=hidden_sizes[:-1],
        activation=activation,
        output_activation=activation,
        device=device,
        dtype=dtype,
        ortho_init=True,
        output_gain=nn.init.calculate_gain("relu"),
    )

    # ── Actor head (single Linear → NormalParamExtractor) ─────────────
    actor_head = make_mlp(
        in_features=hidden_sizes[-1],
        out_features=2 * act_dim,
        hidden_sizes=[],
        device=device,
        dtype=dtype,
        ortho_init=True,
        output_gain=0.01,
    )

    # ── Critic head (single Linear → 1) ──────────────────────────────
    critic_head = make_mlp(
        in_features=hidden_sizes[-1],
        out_features=1,
        hidden_sizes=[],
        device=device,
        dtype=dtype,
        ortho_init=True,
        output_gain=1.0,
    )

    # Full nets — trunk_net is the SAME object → weights are shared
    actor_net = nn.Sequential(trunk_net, actor_head, NormalParamExtractor())
    critic_net = nn.Sequential(trunk_net, critic_head)

    # ── TorchRL wrappers (identical API to make_actor / make_critic) ──
    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
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
    critic = ValueOperator(module=critic_net, in_keys=["observation"])

    return actor, critic


def make_ppo_models(
    env: EnvBase,
    cfg: DictConfig,
    device: torch.device | str = "cpu",
    network_device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> dict:
    """Construct actor, critic, advantage estimator, loss, and optimizer.

    Args:
        env: A TorchRL environment (used for specs).
        cfg: Flat Hydra config that must contain the keys listed in
            ``config/``.
        device: Device for environment/collector (used for data transfers).
        network_device: Device for neural networks. Defaults to ``device``
            if not specified, or uses ``cfg.agent.network_device`` if available.
        dtype: Optional dtype for network parameters (e.g., torch.bfloat16).

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
    share_trunk = cfg.agent.get("share_trunk", False)
    compile_enabled = cfg.agent.get("compile", False)
    compile_mode = cfg.agent.get("compile_mode", "default") if compile_enabled else None

    if share_trunk:
        log.info("Using shared actor-critic trunk")
        actor, critic = make_shared_actor_critic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            device=network_device,
            action_spec=env.action_spec_unbatched,
            dtype=dtype,
        )
    else:
        actor = make_actor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            device=network_device,
            action_spec=env.action_spec_unbatched,
            dtype=dtype,
        )
        critic = make_critic(
            obs_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            device=network_device,
            dtype=dtype,
        )

    if compile_enabled:
        log.info("Compiling network modules with torch.compile...")
        actor.module[0] = torch.compile(actor.module[0], mode=compile_mode)
        critic.module = torch.compile(critic.module, mode=compile_mode)

    advantage = GAE(
        gamma=cfg.agent.gamma,
        lmbda=cfg.agent.gae_lambda,
        value_network=critic,
        average_gae=False,
    )

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.agent.clip_epsilon,
        entropy_bonus=cfg.agent.entropy_coeff > 0,
        entropy_coeff=cfg.agent.entropy_coeff,
        critic_coeff=cfg.agent.critic_coeff,
        loss_critic_type=cfg.agent.loss_critic_type,
        normalize_advantage=True,
    )

    # Deduplicate params (shared trunk params appear in both actor & critic)
    optim_params = list({id(p): p for p in loss_module.parameters()}.values())

    # -- Optimizer & LR scheduler ---------------------------------------------
    optimizer_type = cfg.agent.get("optimizer", "adamw").lower()

    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(optim_params, lr=cfg.agent.lr)
    elif optimizer_type == "rmsprop":
        optimizer = torch.optim.RMSprop(optim_params, lr=cfg.agent.lr)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    total_iters = cfg.collector.total_frames // cfg.collector.frames_per_batch
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=max(cfg.agent.lr_min / cfg.agent.lr, 1e-8),
        total_iters=total_iters,
    )

    return {
        "actor": actor,
        "advantage": advantage,
        "loss_module": loss_module,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "optim_params": optim_params,
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
        obs_rms: Optional running-mean-std object (or dict with ``mean`` and
            ``std`` keys) used to normalise raw observations before the actor.
    """

    def __init__(
        self,
        actor: ProbabilisticActor,
        device: torch.device | str = "cpu",
        deterministic: bool = True,
        obs_rms: Any | None = None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.device = torch.device(device)
        self.deterministic = deterministic

        # Store obs normalisation stats (mean / std tensors)
        if obs_rms is not None:
            if hasattr(obs_rms, "mean"):  # RunningMeanStd object
                self._obs_mean = obs_rms.mean.clone().to(self.device)
                self._obs_std = obs_rms.std.clone().to(self.device)
            elif isinstance(obs_rms, dict):  # loaded from checkpoint
                self._obs_mean = obs_rms["mean"].to(self.device)
                self._obs_std = obs_rms["std"].to(self.device)
            else:
                self._obs_mean = None
                self._obs_std = None
        else:
            self._obs_mean = None
            self._obs_std = None

    def get_action(self, obs: Dict) -> np.ndarray:
        state = obs["state"]
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        # Apply observation normalisation if stats are available
        if self._obs_mean is not None:
            state_t = (state_t - self._obs_mean) / self._obs_std

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
        obs_rms = checkpoint.get("obs_rms", None)
        return cls(actor=actor, device=device, deterministic=deterministic, obs_rms=obs_rms)

    def save(self, path: str) -> None:
        """Persist the actor (and obs normalisation stats) so it can be loaded later."""
        ckpt: dict[str, Any] = {"actor": self.actor}
        if self._obs_mean is not None:
            ckpt["obs_rms"] = {"mean": self._obs_mean.cpu(), "std": self._obs_std.cpu()}
        torch.save(ckpt, path)

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
