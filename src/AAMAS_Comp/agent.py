from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import torch
from torch.distributions import Normal
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from AAMAS_Comp.base_agent import ModelFreeAgent, ModelBasedAgent

# ── Official ObGD ─────────────────────────────────────────────────────────────
# Taken from: https://github.com/mohmdelsayed/streaming-drl/blob/main/optim.py
class ObGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        super(ObGD, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)
                z_sum += e.abs().sum().item()

        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        step_size = group["lr"] / dot_product if dot_product > 1 else group["lr"]

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                p.data.add_(delta * e, alpha=-step_size)
                if reset:
                    e.zero_()


# ── Programmatic reward functions ─────────────────────────────────────────────

def compute_reward_ant(obs_state, prev_obs_state, action):
    # reward = healthy_reward + forward_reward - ctrl_cost - contact_cost.
    torso_z        = float(obs_state[0])
    # healthy if state space values are finite and z-coordinate of torso
    # is in [0.2, 1.0]
    healthy_reward = 1.0 if 0.2 <= torso_z <= 1.0 else 0.0
    # forward reward, positive if Ant moves forward in x-direction
    # idx=13 dives x-coord velocity of the torso
    # the froward_reward_weigth is by default 1 hence it is 1 here
    forward_reward = float(obs_state[13])
    # ctrl_cost is a negative reward that penalizes the Ant for taking
    # actions that are too large, the weight is by default 0.5
    # quantified by euclidian norm of the action vector
    ctrl_cost      = 0.5 * float(np.sum(action ** 2))
    # if len(obs_state) >= 105:
        # in v4 of the environment use_contract-force is False and this
        # does not exist
    #     cfrc_clip    = np.clip(obs_state[27:105], -1.0, 1.0)
    #     contact_cost = 5e-4 * float(np.sum(cfrc_clip ** 2))
    # else:
      # contact_cost = 0.0
    total_reward = healthy_reward + forward_reward - ctrl_cost # - contact_cost

    # print("computed healthy reward", healthy_reward)
    # print("computed forward reward", forward_reward)
    # print("computed ctrl_cost", ctrl_cost)
    # print("computed contact_cost", contact_cost)
    # print("computed total_reward", total_reward)

    return total_reward

def compute_reward_cartpole(obs_state):
    terminated = abs(float(obs_state[0])) > 2.4 or abs(float(obs_state[2])) > 0.2094
    return 0.0 if terminated else 1.0

def compute_reward_frozenlake(obs_state, grid_size=4):
    position = int(np.argmax(obs_state)) if hasattr(obs_state, '__len__') \
               else int(obs_state)
    return 1.0 if position == grid_size * grid_size - 1 else 0.0

REWARD_FNS = {
    "Ant-v5":        compute_reward_ant,
    "CartPole-v1":   compute_reward_cartpole,
    "FrozenLake-v1": compute_reward_frozenlake,
}


class MyModelFreeAgent(ModelFreeAgent):
    """
    Adaptive agent mirroring StreamAC from the paper as closely as possible:
      - Uses the official ObGD optimizer class directly on actor/critic params
      - PPO/PLR pre-trained weights loaded from checkpoint
      - Entropy regularization: τ · sign(δ) · H(π(·|s)), identical to paper
      - EWC penalty injected into actor .grad before ObGD accumulates trace
      - Proper episode boundary handling: prev_state=None after done
    """

    def __init__(
        self,
        model_path: str,
        env_id: str,
        device: str = "cpu",
        lam: float            = 0.8,
        gamma: float          = 0.99,
        lr: float             = 1.0, # 1.0,
        kappa_pi: float       = 3.0,
        kappa_v: float        = 2.0,
        entropy_coeff: float  = 0.01,
        use_ewc: bool         = True,
        ewc_lambda: float     = 500.0,
        online_learning: bool = True,
        deterministic: bool   = False,
    ):
        super().__init__()
        self.env_id          = env_id
        self.device          = torch.device(device)
        self.gamma           = gamma
        self.lam             = lam
        self.entropy_coeff   = entropy_coeff
        self.use_ewc         = use_ewc
        self.ewc_lambda      = ewc_lambda
        self.online_learning = online_learning
        self.deterministic   = deterministic

        ckpt = torch.load(model_path, map_location=device, weights_only=False)

        # ── Actor & critic ───────────────────────────────────────────────────
        self._actor = ckpt["actor"].to(self.device)
        self._actor.train()

        if "critic" in ckpt:
            self._critic = ckpt["critic"].to(self.device)
            self._critic.train()
        else:
            self._critic = None
            if online_learning:
                import warnings
                warnings.warn(
                    "No 'critic' key in checkpoint — online learning disabled.",
                    RuntimeWarning,
                )
                self.online_learning = False

        # ── Obs normalisation ────────────────────────────────────────────────
        obs_rms = ckpt.get("obs_rms", None)
        if obs_rms is not None:
            self._obs_mean = obs_rms["mean"].to(self.device).float()
            self._obs_std  = obs_rms["std"].to(self.device).float()
        else:
            self._obs_mean = None
            self._obs_std  = None

        # ── EWC: freeze θ* and Fisher ────────────────────────────────────────
        if self.use_ewc:
            self._theta_star = {
                n: p.detach().clone()
                for n, p in self._actor.named_parameters()
            }
            self._fisher = ckpt.get("fisher", {
                n: torch.ones_like(p) for n, p in self._theta_star.items()
            })
        else:
            self._theta_star = {}
            self._fisher     = {}

        # ── ObGD optimizers ──────────────────────────────────────────────────
        if self.online_learning:
            self.optimizer_policy = ObGD(
                self._actor.parameters(),
                lr=lr, gamma=gamma, lamda=lam, kappa=kappa_pi,
            )
            self.optimizer_value = ObGD(
                self._critic.parameters(),
                lr=lr, gamma=gamma, lamda=lam, kappa=kappa_v,
            )

        # ── Transition cache ─────────────────────────────────────────────────
        self._prev_state:  Optional[torch.Tensor] = None
        self._prev_action: Optional[torch.Tensor] = None
        self._prev_relative_time = None

    def _normalise(self, state: np.ndarray) -> torch.Tensor:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if self._obs_mean is not None and self._obs_mean.shape == s.shape:
            s = (s - self._obs_mean) / (self._obs_std + 1e-8)
        return s

    def _compute_reward(self, state, prev_state, action):
        fn = REWARD_FNS.get(self.env_id)
        if fn is None:
            return 0.0
        if self.env_id == "Ant-v5":
            return fn(state, prev_state, action)
        return fn(state)

    def _critic_value(self, s: torch.Tensor) -> torch.Tensor:
        """V(s) — squeezes [1,1] → scalar tensor, keeps grad."""
        td = TensorDict({"observation": s.unsqueeze(0)}, batch_size=[1],
                        device=self.device)
        td = self._critic(td)
        return td["state_value"].squeeze()

    def _actor_forward(self, s: torch.Tensor,
                       a: torch.Tensor) -> tuple[torch.Tensor, Normal]:
        """
        Single actor forward pass that returns both log π(a|s) and the
        distribution.

        The actor writes 'loc' and 'scale',
        so we reconstruct Normal(loc, scale) here — identical to what the
        official StreamAC does with its Categorical/Normal dist object.

        Returns:
            log_prob : scalar tensor, log π(a|s)
            dist     : Normal distribution at s, used for entropy
        """
        td = TensorDict(
            {"observation": s.unsqueeze(0), "action": a.unsqueeze(0)},
            batch_size=[1], device=self.device,
        )
        with set_exploration_type(ExplorationType.RANDOM):
            td = self._actor(td)

        log_prob = td["action_log_prob"].squeeze()          # scalar
        dist     = Normal(td["loc"].squeeze(0),             # reconstruct from
                          td["scale"].squeeze(0))           # confirmed keys
        return log_prob, dist

    def _stream_update(self, s: torch.Tensor, a: torch.Tensor,
                       r: float, s_prime: torch.Tensor, done: bool):
        """
        Mirrors StreamAC.update_params exactly, including entropy:

            delta          = r + γ·V(s') - V(s)

            log_prob_pi    = -(log π(a|s)).sum()
            entropy_pi     = -τ · sign(δ) · H(π(·|s))    <- paper Appendix E
            value_output   = -V(s)

            optimizer_value.zero_grad();  optimizer_policy.zero_grad()
            value_output.backward()
            (log_prob_pi + entropy_pi).backward()

            [EWC: add F·(θ-θ*) to actor .grad]

            optimizer_policy.step(delta, reset=done)
            optimizer_value.step(delta, reset=done)
        """
        done_mask = 0.0 if done else 1.0

        # TD error
        v_s     = self._critic_value(s)
        v_prime = self._critic_value(s_prime)
        td_target = r + self.gamma * v_prime.detach() * done_mask
        delta     = (td_target - v_s).detach().item()

        # Policy: log prob + entropy
        # Mirrors:  log_prob_pi = -(dist.log_prob(a)).sum()
        #           entropy_pi  = -entropy_coeff * dist.entropy().sum() * sign(δ)
        #           (log_prob_pi + entropy_pi).backward()
        log_prob, dist = self._actor_forward(s, a)
        # Check sign of log_prob
        log_prob_pi = -log_prob.sum() 
        entropy_pi  = (-self.entropy_coeff
                       * dist.entropy().sum()
                       * torch.sign(torch.tensor(delta)).item())

        value_output = -v_s

        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()

        value_output.backward()
        (log_prob_pi + entropy_pi).backward()

        # EWC: add F·(θ-θ*) to actor .grad before ObGD reads it
        if self.use_ewc:
            with torch.no_grad():
                for name, param in self._actor.named_parameters():
                    if param.grad is not None and name in self._fisher:
                        param.grad.add_(
                            self.ewc_lambda
                            * self._fisher[name]
                            * (param - self._theta_star[name])
                        )

        # ObGD steps
        self.optimizer_policy.step(delta, reset=done)
        self.optimizer_value.step(delta, reset=done)


    def get_action(self, obs: Dict) -> np.ndarray:
        raw_state  = obs["state"]        if isinstance(obs, dict) else obs
        relative_time = obs.get("relative_time", None) if isinstance(obs, dict) else None

        # Episodes is finished if the relative time resets back to 0
        done = (
            self._prev_relative_time is not None
            and relative_time is not None
            and relative_time < self._prev_relative_time
        )
        self._prev_relative_time = relative_time
        s = self._normalise(raw_state)

        if self.online_learning and self._prev_state is not None:
            r = self._compute_reward(
                state      = raw_state,
                prev_state = self._prev_state.cpu().numpy(),
                action     = self._prev_action.cpu().numpy(),
            )
            self._stream_update(
                s       = self._prev_state,
                a       = self._prev_action,
                r       = r,
                s_prime = s,
                done    = done,   # passes reset=True into ObGD when episode ended
            )
            if done:
                # clear cache so next call doesn't cross episode boundary
                self._prev_state  = None
                self._prev_action = None
                return self._sample_action(s)

        action = self._sample_action(s)
        self._prev_state  = s.detach()
        self._prev_action = torch.as_tensor(
            action, dtype=torch.float32, device=self.device
        ).detach()
        return action

    def _sample_action(self, s: torch.Tensor) -> np.ndarray:
        td = TensorDict({"observation": s.unsqueeze(0)}, batch_size=[1],
                        device=self.device)
        explore = ExplorationType.DETERMINISTIC if self.deterministic \
                else ExplorationType.RANDOM
        with set_exploration_type(explore), torch.no_grad():
            td = self._actor(td)
        return td["action"].squeeze(0).cpu().numpy()

    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)


# ── Fisher computation ────────────────────────────────────────────────────────

def compute_and_save_fisher(
    model_path, output_path, actor, tensordict_data,
    device="cpu", n_samples=2000,
):
    obs = tensordict_data["observation"].reshape(-1, tensordict_data["observation"].shape[-1])[:n_samples]
    act = tensordict_data["action"].reshape(-1, tensordict_data["action"].shape[-1])[:n_samples]
    N   = obs.shape[0]
    fisher = {n: torch.zeros_like(p) for n, p in actor.named_parameters()}
    actor.train()
    for t in range(N):
        actor.zero_grad()
        td_t = TensorDict(
            {"observation": obs[t].unsqueeze(0), "action": act[t].unsqueeze(0)},
            batch_size=[1], device=device,
        )
        with set_exploration_type(ExplorationType.RANDOM):
            td_t = actor(td_t)
        td_t["action_log_prob"].sum().backward()
        for n, p in actor.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
    for n in fisher:
        fisher[n] /= N
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    ckpt["fisher"]     = fisher
    ckpt["theta_star"] = {n: p.detach().clone() for n, p in actor.named_parameters()}
    torch.save(ckpt, output_path)
    print(f"EWC checkpoint saved → {output_path}  ({N} samples)")


class MyModelBasedAgent(ModelBasedAgent):
    def __init__(self):
        raise NotImplementedError
    def get_action(self, obs, planning_env):
        raise NotImplementedError
    def set_seed(self, seed):
        raise NotImplementedError
