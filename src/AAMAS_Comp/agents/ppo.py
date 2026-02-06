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
    )
    critic = ValueOperator(module=net, in_keys=["observation"])
    
    return critic


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

    # -- Apply torch.compile if enabled -------------------------------------
    # Compile the underlying neural network modules BEFORE creating loss module
    # This ensures the loss module captures the compiled versions
    if cfg.agent.get("compile", False):
        log.info("Compiling network modules with torch.compile...")
        compile_mode = cfg.agent.get("compile_mode", "default")
        # Compile the underlying Sequential modules, not the TensorDictModule wrappers
        # Structure is: actor.module (TensorDictModule) -> [0] (Sequential with MLP + NormalParamExtractor)
        actor.module[0] = torch.compile(actor.module[0], mode=compile_mode)
        # Critic has ValueOperator wrapping TensorDictModule wrapping the MLP
        critic.module = torch.compile(critic.module, mode=compile_mode)

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
    # Muon only supports 2D parameters, so we separate them
    params_2d = [p for p in loss_module.parameters() if p.dim() >= 2]
    params_1d = [p for p in loss_module.parameters() if p.dim() < 2]
    
    # Create separate optimizers for 2D and 1D parameters
    class CombinedOptimizer(torch.optim.Optimizer):
        """Wrapper to handle both Muon (2D) and AdamW (1D) optimizers together."""
        def __init__(self, muon_opt, adamw_opt):
            self.muon_opt = muon_opt
            self.adamw_opt = adamw_opt
            # Combine param groups from both optimizers
            self.param_groups = muon_opt.param_groups + adamw_opt.param_groups
            self.defaults = {}
            self._step = self._create_step()
        
        def _create_step(self):
            """Create the compiled step function."""
            def step_fn():
                self.muon_opt.step()
                self.adamw_opt.step()
            return torch.compile(step_fn, backend="eager")
        
        def step(self, closure=None):
            self._step()
        
        def zero_grad(self, set_to_none=False):
            self.muon_opt.zero_grad(set_to_none=set_to_none)
            self.adamw_opt.zero_grad(set_to_none=set_to_none)
    
    # Create optimizer based on config
    optimizer_type = cfg.agent.get("optimizer", "adamw").lower()
    
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(loss_module.parameters(), lr=cfg.agent.lr)
    elif optimizer_type == "rmsprop":
        optimizer = torch.optim.RMSprop(loss_module.parameters(), lr=cfg.agent.lr)
    elif optimizer_type == "muon":
        if not params_2d:
            raise ValueError("Muon optimizer requires 2D parameters, but none found")
        optimizer = torch.optim.Muon(params_2d, lr=cfg.agent.lr)
    elif optimizer_type == "muon_adamw":
        muon_opt = torch.optim.Muon(params_2d, lr=cfg.agent.lr) if params_2d else None
        adamw_opt = torch.optim.AdamW(params_1d, lr=cfg.agent.lr) if params_1d else None
        
        if muon_opt and adamw_opt:
            optimizer = CombinedOptimizer(muon_opt, adamw_opt)
        elif muon_opt:
            optimizer = muon_opt
        else:
            optimizer = adamw_opt
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

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
