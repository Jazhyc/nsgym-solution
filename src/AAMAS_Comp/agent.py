"""
Adaptive NS-Gym agent: PPO/PLR prior + online stream AC(λ) + EWC anchoring.

Strategy
--------
1. Load a pre-trained PPO checkpoint (trained with PLR over NS configs).
2. At eval time, reconstruct rewards programmatically from obs["state"]
   (reward functions for Ant, CartPole, FrozenLake are fully deterministic
   from the state — no need for the env to pass them).
3. Do a one-step-delayed stream AC(λ) update with ObGD after each transition.
4. EWC penalty anchors updates toward the PLR prior θ* so the general policy
   is not overwritten by local adaptation.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from AAMAS_Comp.base_agent import ModelFreeAgent, ModelBasedAgent
from AAMAS_Comp.agents.ppo import PPOAgent as TorchRLPPOAgent


# ── Programmatic reward functions ────────────────────────────────────────────

def compute_reward_ant(obs_state: np.ndarray,
                       prev_obs_state: Optional[np.ndarray],
                       action: np.ndarray) -> float:
    """
    Ant-v5 reward (gymnasium AntEnv defaults):
        r = forward_reward + healthy_reward - ctrl_cost

    Default 27-dim obs layout:
        [0]      torso z position
        [1:13]   joint positions / quaternion
        [13:27]  velocities; [13] = x-velocity (forward)
    """
    torso_z        = float(obs_state[0])
    healthy_reward = 1.0 if 0.2 <= torso_z <= 1.0 else 0.0
    x_velocity     = float(obs_state[13]) if len(obs_state) > 13 else 0.0
    ctrl_cost      = 0.5 * float(np.sum(action ** 2))
    return x_velocity + healthy_reward - ctrl_cost


def compute_reward_cartpole(obs_state: np.ndarray) -> float:
    """
    CartPole-v1: +1 every step while not terminated.
    obs = [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    """
    terminated = abs(float(obs_state[0])) > 2.4 or abs(float(obs_state[2])) > 0.2094
    return 0.0 if terminated else 1.0


def compute_reward_frozenlake(obs_state: np.ndarray, grid_size: int = 4) -> float:
    """
    FrozenLake-v1: +1 if on goal tile (bottom-right corner of grid).
    obs is a one-hot vector of length grid_size*grid_size.
    """
    position = int(np.argmax(obs_state)) if hasattr(obs_state, '__len__') \
               else int(obs_state)
    return 1.0 if position == grid_size * grid_size - 1 else 0.0


REWARD_FNS = {
    "Ant-v5":        compute_reward_ant,
    "CartPole-v1":   compute_reward_cartpole,
    "FrozenLake-v1": compute_reward_frozenlake,
}


# ── ObGD optimiser ────────────────────────────────────────────────────────────

def obgd_update(params, traces, delta: float, alpha: float, kappa: float):
    """
    Overshooting-bounded Gradient Descent (Elsayed et al. 2024, Alg. 3).
        α' = min(α / M, α),  M = α · κ · max(|δ|, 1) · ‖z‖₁
        w  ← w + α' · δ · z
    """
    delta_bar = max(abs(delta), 1.0)
    z_l1 = sum(z.abs().sum().item() for z in traces)
    if z_l1 == 0.0:
        return
    M         = alpha * kappa * delta_bar * z_l1
    alpha_eff = min(alpha / M, alpha)
    with torch.no_grad():
        for p, z in zip(params, traces):
            p.add_(alpha_eff * delta * z)


# ── Main agent ────────────────────────────────────────────────────────────────

class MyModelFreeAgent(ModelFreeAgent):
    """
    Adaptive agent:
      - PPO/PLR pre-trained actor + critic loaded from checkpoint
      - Online stream AC(λ) updates via ObGD (one-step delayed)
      - Programmatic reward reconstruction
      - EWC anchoring to PLR prior θ* (optional, use_ewc=True/False)
    """

    def __init__(
        self,
        model_path: str,
        env_id: str,
        device: str = "cpu",
        # Stream AC (Elsayed et al. 2024 defaults)
        lam: float            = 0.8,
        gamma: float          = 0.99,
        alpha_pi: float       = 1.0,
        alpha_v: float        = 1.0,
        kappa_pi: float       = 3.0,
        kappa_v: float        = 2.0,
        # EWC
        use_ewc: bool         = True,
        ewc_lambda: float     = 1,
        # Adaptation window after a detected change
        adapt_window: int     = 50,
        kappa_pi_adapt: float = 1.2,
        kappa_v_adapt: float  = 0.8,
        online_learning: bool = True,
        deterministic: bool   = False,
    ):
        super().__init__()
        self.env_id          = env_id
        self.device          = torch.device(device)
        self.lam             = lam
        self.gamma           = gamma
        self.alpha_pi        = alpha_pi
        self.alpha_v         = alpha_v
        self.kappa_pi_base   = kappa_pi
        self.kappa_v_base    = kappa_v
        self.kappa_pi_adapt  = kappa_pi_adapt
        self.kappa_v_adapt   = kappa_v_adapt
        self.use_ewc         = use_ewc
        self.ewc_lambda      = ewc_lambda
        self.adapt_window    = adapt_window
        self.online_learning = online_learning
        self.deterministic   = deterministic

        ckpt = torch.load(model_path, map_location=device, weights_only=False)

        # Actor: ProbabilisticActor (TorchRL), handles sampling + log_prob
        self._actor = ckpt["actor"].to(self.device)
        self._actor.train()

        # Critic: ValueOperator (TorchRL). Falls back to disabling
        # online learning gracefully.
        if "critic" in ckpt:
            self._critic = ckpt["critic"].to(self.device)
            self._critic.train()
        else:
            self._critic = None
            if online_learning:
                import warnings
                warnings.warn(
                    "No 'critic' key in checkpoint — online learning disabled.\n"
                    "Fix: PPOAgent.save(path, critic=critic) in train.py.",
                    RuntimeWarning,
                )
                self.online_learning = False

        # Obs normalisation
        obs_rms = ckpt.get("obs_rms", None)
        if obs_rms is not None:
            self._obs_mean = obs_rms["mean"].to(self.device).float()
            self._obs_std  = obs_rms["std"].to(self.device).float()
        else:
            self._obs_mean = None
            self._obs_std  = None

        # EWC: freeze θ* and Fisher (actor only)
        if self.use_ewc:
            self._theta_star = {
                n: p.detach().clone()
                for n, p in self._actor.named_parameters()
            }
            # Fisher from checkpoint if compute_and_save_fisher was called;
            # otherwise fall back to uniform weights (= plain L2 penalty toward θ*)
            self._fisher = ckpt.get("fisher", {
                n: torch.ones_like(p) for n, p in self._theta_star.items()
            })
        else:
            self._theta_star = {}
            self._fisher     = {}

        # Eligibility traces
        self._z_actor  = [torch.zeros_like(p)
                          for p in self._actor.parameters()]
        self._z_critic = [torch.zeros_like(p)
                          for p in self._critic.parameters()] \
                         if self._critic is not None else []

        # Transition cache (one-step delay)
        self._prev_state:  Optional[torch.Tensor] = None
        self._prev_action: Optional[torch.Tensor] = None

        # Adaptation state
        self._adapt_steps = 0

    @property
    def _kappa_pi(self) -> float:
        return self.kappa_pi_adapt if self._adapt_steps > 0 else self.kappa_pi_base

    @property
    def _kappa_v(self) -> float:
        return self.kappa_v_adapt if self._adapt_steps > 0 else self.kappa_v_base

    def _reset_traces(self):
        self._z_actor  = [torch.zeros_like(z) for z in self._z_actor]
        self._z_critic = [torch.zeros_like(z) for z in self._z_critic]

    def _normalise(self, state: np.ndarray) -> torch.Tensor:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if self._obs_mean is not None and self._obs_mean.shape == s.shape:
            s = (s - self._obs_mean) / (self._obs_std + 1e-8)
        return s

    def _compute_reward(self, state: np.ndarray,
                        prev_state: Optional[np.ndarray],
                        action: np.ndarray) -> float:
        fn = REWARD_FNS.get(self.env_id)
        if fn is None:
            return 0.0
        if self.env_id == "Ant-v5":
            return fn(state, prev_state, action)
        return fn(state)

    def _critic_value(self, s: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ValueOperator.
        ValueOperator expects TensorDict with key "observation" and writes
        "state_value".
        """
        td = TensorDict({"observation": s.unsqueeze(0)}, batch_size=[1],
                        device=self.device)
        td = self._critic(td)
        return td["state_value"].squeeze()

    def _actor_log_prob(self, s: torch.Tensor,
                        a: torch.Tensor) -> torch.Tensor:
        """
        Log-probability of action a under current policy at state s.
        ProbabilisticActor with return_log_prob=True writes "action_log_prob"
        to the output TensorDict.
        When "action" is present in the input TD, ProbabilisticActor uses it
        directly instead of sampling, so we get log π(a|s) for the stored a.
        """
        td = TensorDict(
            {"observation": s.unsqueeze(0), "action": a.unsqueeze(0)},
            batch_size=[1],
            device=self.device,
        )
        with set_exploration_type(ExplorationType.RANDOM):
            td = self._actor(td)
        return td["action_log_prob"].squeeze()

    def _stream_update(self, s: torch.Tensor, a: torch.Tensor,
                       r: float, s_next: torch.Tensor, done: bool):
        """
        One stream AC(λ) + EWC step.

        δ   = r + γ·V(s') - V(s)
        z_θ ← γλ·z_θ + ∇_θ log π(a|s)   (EWC correction applied to trace if enabled)
        z_w ← γλ·z_w + ∇_w V(s)
        θ   ← ObGD(z_θ, δ, α_π, κ_π)
        w   ← ObGD(z_w, δ, α_v, κ_v)
        """
        self._actor.zero_grad()
        self._critic.zero_grad()

        # TD error
        v      = self._critic_value(s)
        v_next = self._critic_value(s_next).detach() if not done \
                 else torch.tensor(0.0, device=self.device)
        delta  = float(r + self.gamma * v_next - v.detach())

        # Actor trace
        log_prob = self._actor_log_prob(s, a)
        log_prob.backward(retain_graph=True)
        self._z_actor = [
            self.gamma * self.lam * zt +
            (p.grad.detach() if p.grad is not None else torch.zeros_like(p))
            for zt, p in zip(self._z_actor, self._actor.parameters())
        ]

        # EWC correction: subtract λ·F·(θ-θ*) from actor trace so that
        # ObGD implicitly applies the EWC penalty without a second backward.
        if self.use_ewc:
            with torch.no_grad():
                for (name, param), z in zip(self._actor.named_parameters(),
                                            self._z_actor):
                    if name in self._fisher:
                        ewc_g = (self.ewc_lambda * self._fisher[name]
                                 * (param - self._theta_star[name]))
                        z.sub_(ewc_g / (abs(delta) + 1e-8))

        # Critic trace
        self._actor.zero_grad()
        self._critic.zero_grad()
        v.backward()
        self._z_critic = [
            self.gamma * self.lam * zw +
            (p.grad.detach() if p.grad is not None else torch.zeros_like(p))
            for zw, p in zip(self._z_critic, self._critic.parameters())
        ]

        # ObGD parameter updates
        obgd_update(self._actor.parameters(),  self._z_actor,
                    delta, self.alpha_pi, self._kappa_pi)
        obgd_update(self._critic.parameters(), self._z_critic,
                    delta, self.alpha_v,  self._kappa_v)

        if done:
            self._reset_traces()


    def get_action(self, obs: Dict) -> np.ndarray:
        raw_state    = obs["state"] if isinstance(obs, dict) else obs
        env_changed  = obs.get("env_change",  False) if isinstance(obs, dict) else False
        delta_change = obs.get("delta_change", None) if isinstance(obs, dict) else None

        s = self._normalise(raw_state)

        # Notification-aware adaptation
        if env_changed:
            """
            Sketch of what could be done, disabled for now
            self._reset_traces()
            self._adapt_steps = self.adapt_window
            if delta_change is not None:
                magnitude = max(abs(v) for v in delta_change.values()) \
                            if isinstance(delta_change, dict) \
                            else abs(float(delta_change))
                self._adapt_steps = 0 if magnitude < 0.05 else int(
                    self.adapt_window * min(magnitude, 3.0)
                )
             """

        if self._adapt_steps > 0:
            self._adapt_steps -= 1

        # One-step-delayed stream update
        if self.online_learning and self._prev_state is not None:
            prev_np = self._prev_action.cpu().numpy()
            r = self._compute_reward(
                state      = raw_state,
                prev_state = self._prev_state.cpu().numpy(),
                action     = prev_np,
            )
            self._stream_update(
                s      = self._prev_state,
                a      = self._prev_action,
                r      = r,
                s_next = s,
                done   = False,
            )

        # Sample action and save the selected state and action to be used
        # for learning update later
        td = TensorDict({"observation": s.unsqueeze(0)}, batch_size=[1],
                        device=self.device)
        explore = ExplorationType.DETERMINISTIC if self.deterministic \
                  else ExplorationType.RANDOM
        with set_exploration_type(explore), torch.no_grad():
            td = self._actor(td)
        action = td["action"].squeeze(0).cpu().numpy()

        self._prev_state  = s.detach()
        self._prev_action = torch.as_tensor(
            action, dtype=torch.float32, device=self.device
        ).detach()
        return action

    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)


# ── Fisher computation (run once post-training, called from train.py) ────────

def compute_and_save_fisher(
    model_path: str,
    output_path: str,
    actor,            # live ProbabilisticActor from training
    tensordict_data,  # last collect iteration tensordict (already normalised)
    device: str = "cpu",
    n_samples: int = 2000,
):
    """
    Compute diagonal Fisher from the last training batch and bake into checkpoint.
    TODO: Double check the computation later, there are some nuances
    to estimating the FIM

    Call in train.py after the training loop, before wandb.finish():

        if cfg.agent.get("compute_fisher", False):
            from AAMAS_Comp.agent import compute_and_save_fisher
            compute_and_save_fisher(
                model_path      = str(final_path),
                output_path     = str(ckpt_dir / "ppo_final_ewc.pt"),
                actor           = actor,
                tensordict_data = tensordict_data,
                device          = str(device),
                n_samples       = cfg.agent.get("fisher_samples", 2000),
            )
    """
    fisher = {n: torch.zeros_like(p) for n, p in actor.named_parameters()}

    obs = tensordict_data["observation"]
    act = tensordict_data["action"]

    obs = obs.reshape(-1, obs.shape[-1])[:n_samples]
    act = act.reshape(-1, act.shape[-1])[:n_samples]
    N   = obs.shape[0]

    fisher = {n: torch.zeros_like(p) for n, p in actor.named_parameters()}

    actor.train()
    for t in range(N):
        actor.zero_grad()
        td_t = TensorDict(
            {"observation": obs[t].unsqueeze(0),
             "action":      act[t].unsqueeze(0)},
            batch_size=[1],
            device=device,
        )
        with set_exploration_type(ExplorationType.RANDOM):
            td_t = actor(td_t)
        log_prob = td_t["action_log_prob"].sum()
        log_prob.backward()
        for n, p in actor.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2

    for n in fisher:
        fisher[n] /= N

    theta_star = {n: p.detach().clone() for n, p in actor.named_parameters()}

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    ckpt["fisher"]     = fisher
    ckpt["theta_star"] = theta_star
    torch.save(ckpt, output_path)
    print(f"EWC checkpoint saved → {output_path}  ({N} samples)")


class MyModelBasedAgent(ModelBasedAgent):
    def __init__(self):
        raise NotImplementedError

    def get_action(self, obs: Dict, planning_env):
        raise NotImplementedError

    def set_seed(self, seed):
        raise NotImplementedError
