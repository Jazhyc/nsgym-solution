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
from tensordict.nn import TensorDictModule, TensorDictParams
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torch.distributions import OneHotCategorical

from torchrl.data import OneHot
from torchrl.envs import EnvBase
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE, VTrace

from AAMAS_Comp.agents.networks import make_mlp
from AAMAS_Comp.base_agent import ModelFreeAgent

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DiscoPO loss (Lu et al., NeurIPS 2022)
# ---------------------------------------------------------------------------

class DiscoPOLoss(ClipPPOLoss):
    """Discovered Policy Optimization surrogate objective.

    Replaces the clipped PPO actor loss with the piecewise surrogate
    discovered by Lu et al. (NeurIPS 2022)::

        f(r, A) = ReLU((r-1)*A - α*tanh((r-1)*A/α))   if A >= 0
                  ReLU(log(r)*A - β*tanh(log(r)*A/β))  if A <  0

    The critic loss, entropy bonus, and advantage normalisation are
    inherited unchanged from ``ClipPPOLoss``.
    """

    # Re-declare annotations so TorchRL's LossModule registers params correctly
    actor_network: "TensorDictModule"
    critic_network: "TensorDictModule"
    actor_network_params: "TensorDictParams"
    critic_network_params: "TensorDictParams"
    target_actor_network_params: "TensorDictParams"
    target_critic_network_params: "TensorDictParams"

    def __init__(self, *args, disco_alpha: float = 2.0, disco_beta: float = 0.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.disco_alpha = disco_alpha
        self.disco_beta = disco_beta

    def forward(self, tensordict):
        from tensordict import TensorDict
        from torchrl.objectives.utils import _reduce

        tensordict = tensordict.clone(False)

        # ── Advantage (reuse parent's normalisation logic) ────────────
        advantage = tensordict.get(self.tensor_keys.advantage, None, as_padded_tensor=True)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self._cached_critic_network_params_detached,
                target_params=self.target_critic_network_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean()
            scale = advantage.std().clamp_min(1e-8)
            advantage = (advantage - loc) / scale

        # ── Importance-sampling ratio (from parent) ───────────────────
        log_weight, dist, kl_approx = self._log_weight(
            tensordict, adv_shape=advantage.shape[:-1]
        )
        # log_weight shape: (*batch, 1)  — squeeze trailing dim
        log_ratio = log_weight.squeeze(-1)
        ratio = log_ratio.exp()
        adv = advantage.squeeze(-1)

        # ── DiscoPO piecewise surrogate ───────────────────────────────
        # f(r,A) is the clipping penalty; full objective is r*A - f(r,A)
        alpha, beta = self.disco_alpha, self.disco_beta

        # A >= 0 branch: clip = ReLU((r-1)*A - α*tanh((r-1)*A / α))
        u = (ratio - 1.0) * adv
        f_pos = torch.relu(u - alpha * torch.tanh(u / alpha))

        # A < 0 branch:  clip = ReLU(log(r)*A - β*tanh(log(r)*A / β))
        v = log_ratio * adv
        f_neg = torch.relu(v - beta * torch.tanh(v / beta))

        clip_penalty = torch.where(adv >= 0, f_pos, f_neg)
        gain = ratio * adv - clip_penalty

        # ── ESS for logging ───────────────────────────────────────────
        with torch.no_grad():
            lw = log_weight.squeeze()
            ess = (2 * lw.logsumexp(0) - (2 * lw).logsumexp(0)).exp()
            batch = log_weight.shape[0]
            clip_fraction = torch.zeros((), device=gain.device)  # not applicable

        td_out = TensorDict({"loss_objective": -gain})
        td_out.set("clip_fraction", clip_fraction)
        td_out.set("kl_approx", kl_approx.detach().mean())

        # ── Entropy bonus (inherited) ─────────────────────────────────
        if self.entropy_bonus:
            entropy = self._get_entropy(dist, adv_shape=advantage.shape[:-1])
            from tensordict.utils import is_tensor_collection
            if is_tensor_collection(entropy):
                from torchrl.objectives.ppo import _sum_td_features
                td_out.set("composite_entropy", entropy.detach())
                td_out.set("entropy", _sum_td_features(entropy).detach().mean())
            else:
                td_out.set("entropy", entropy.detach().mean())
            td_out.set("loss_entropy", self._weighted_loss_entropy(entropy))

        # ── Critic loss (inherited) ───────────────────────────────────
        if self._has_critic:
            loss_critic, value_clip_fraction, explained_variance = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic)
            if value_clip_fraction is not None:
                td_out.set("value_clip_fraction", value_clip_fraction)
            if explained_variance is not None:
                td_out.set("explained_variance", explained_variance)

        td_out.set("ESS", _reduce(ess, self.reduction) / batch)
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction).squeeze(-1)
            if name.startswith("loss_")
            else value,
        )
        self._clear_weakrefs(
            tensordict, td_out,
            "actor_network_params", "critic_network_params",
            "target_actor_network_params", "target_critic_network_params",
        )
        return td_out


# ---------------------------------------------------------------------------
# RPO (Robust Policy Optimization) changed distribution
# ---------------------------------------------------------------------------

class RPOTanhNormal(TanhNormal):
    """TanhNormal with RPO perturbation of the scale parameter.

    When ``rpo_enabled`` is True (during loss computation), adds
    ``Uniform(0, rpo_alpha)`` noise to the scale before constructing the
    distribution.  This makes the importance-sampling ratio more
    conservative, improving robustness.

    Reference: Liang et al., "RPO: Robust Policy Optimization"
    """

    rpo_alpha: float = 0.5
    rpo_enabled: bool = False

    def __init__(self, loc, scale, *args, **kwargs):
        if RPOTanhNormal.rpo_enabled and RPOTanhNormal.rpo_alpha > 0:
            # Create symmetric noise z ~ U(-alpha, alpha)
            z = (torch.rand_like(loc) * 2 - 1) * RPOTanhNormal.rpo_alpha
            loc = loc + z  # Add noise to mean, not scale
        super().__init__(loc, scale, *args, **kwargs)


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
        distribution_class=RPOTanhNormal,
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


def make_discrete_actor(
    obs_dim: int,
    n_actions: int,
    hidden_sizes: Sequence[int],
    activation: str,
    device: torch.device | str,
    action_spec: Any | None = None,
    dtype: torch.dtype | None = None,
) -> ProbabilisticActor:
    """Build a stochastic Categorical actor for discrete action spaces.

    The network outputs logits of shape ``(n_actions,)`` consumed by a
    ``Categorical`` distribution.  Compatible with ``ClipPPOLoss``.
    """
    mlp = make_mlp(
        in_features=obs_dim,
        out_features=n_actions,
        hidden_sizes=hidden_sizes,
        activation=activation,
        device=device,
        dtype=dtype,
        ortho_init=True,
        output_gain=0.01,
    )
    # Wrap in Sequential so actor.module[0] == mlp (consistent with continuous actor)
    net = nn.Sequential(mlp)

    policy_module = TensorDictModule(
        net,
        in_keys=["observation"],
        out_keys=["logits"],
    )

    actor = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=["logits"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )
    return actor


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
        distribution_class=RPOTanhNormal,
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

    # Detect discrete vs continuous action space.
    # TorchRL wraps gym.spaces.Discrete as OneHot(n) with shape=(n,).
    action_spec_unbatched = env.action_spec_unbatched
    is_discrete_action = isinstance(action_spec_unbatched, OneHot)

    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = action_spec_unbatched.shape[-1]  # n for OneHot, act_dim for continuous
    if is_discrete_action:
        log.info("Discrete action space detected (n=%d)", act_dim)

    hidden_sizes = list(cfg.agent.hidden_sizes)
    activation = cfg.agent.activation
    share_trunk = cfg.agent.get("share_trunk", False)
    compile_enabled = cfg.agent.get("compile", False)
    compile_mode = cfg.agent.get("compile_mode", "default") if compile_enabled else None

    if is_discrete_action:
        if share_trunk:
            log.warning("share_trunk=true is not supported for discrete action spaces; using separate networks")
        actor = make_discrete_actor(
            obs_dim=obs_dim,
            n_actions=act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            device=network_device,
            action_spec=action_spec_unbatched,
            dtype=dtype,
        )
        critic = make_critic(
            obs_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            device=network_device,
            dtype=dtype,
        )
    elif share_trunk:
        log.info("Using shared actor-critic trunk")
        actor, critic = make_shared_actor_critic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            device=network_device,
            action_spec=action_spec_unbatched,
            dtype=dtype,
        )
    else:
        actor = make_actor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            device=network_device,
            action_spec=action_spec_unbatched,
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

    use_vtrace = cfg.agent.get("use_vtrace", False)
    if use_vtrace:
        rho_thresh = cfg.agent.get("vtrace_rho_thresh", 1.0)
        c_thresh = cfg.agent.get("vtrace_c_thresh", 1.0)
        log.info("Using V-trace advantage (rho_thresh=%.1f, c_thresh=%.1f)",
                 rho_thresh, c_thresh)
        advantage = VTrace(
            gamma=cfg.agent.gamma,
            actor_network=actor,
            value_network=critic,
            rho_thresh=rho_thresh,
            c_thresh=c_thresh,
        )
    else:
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

    # Optionally swap in the DiscoPO surrogate objective
    if cfg.agent.get("use_disco", False):
        disco_alpha = cfg.agent.get("disco_alpha", 2.0)
        disco_beta = cfg.agent.get("disco_beta", 0.6)
        log.info("Using DiscoPO loss (alpha=%.2f, beta=%.2f)", disco_alpha, disco_beta)
        loss_module = DiscoPOLoss(
            actor_network=actor,
            critic_network=critic,
            clip_epsilon=cfg.agent.clip_epsilon,
            entropy_bonus=cfg.agent.entropy_coeff > 0,
            entropy_coeff=cfg.agent.entropy_coeff,
            critic_coeff=cfg.agent.critic_coeff,
            loss_critic_type=cfg.agent.loss_critic_type,
            normalize_advantage=True,
            disco_alpha=disco_alpha,
            disco_beta=disco_beta,
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
        "critic": critic,
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
        critic = None,
        device: torch.device | str = "cpu",
        deterministic: bool = True,
        obs_rms: Any | None = None,
        context_meta: dict | None = None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.device = torch.device(device)
        self.deterministic = deterministic

        # Store obs normalisation stats (mean / std tensors)
        if obs_rms is not None:
            if hasattr(obs_rms, "mean"):  # RunningMeanStd object
                self._obs_mean = obs_rms.mean.clone().to(self.device).float()
                self._obs_std = obs_rms.std.clone().to(self.device).float()
            elif isinstance(obs_rms, dict):  # loaded from checkpoint
                self._obs_mean = obs_rms["mean"].to(self.device).float()
                self._obs_std = obs_rms["std"].to(self.device).float()
            else:
                self._obs_mean = None
                self._obs_std = None
        else:
            self._obs_mean = None
            self._obs_std = None

        # Context metadata for inference-time obs reconstruction
        self.context_meta = context_meta or {}

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
        if self.critic is not None:
            ckpt["critic"] = self.critic
        if self._obs_mean is not None:
            ckpt["obs_rms"] = {"mean": self._obs_mean.cpu(), "std": self._obs_std.cpu()}
        if self.context_meta:
            ckpt["context_meta"] = self.context_meta
        torch.save(ckpt, path)

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
