"""Profile per-step policy latency across cumulative inference optimisation stages.

Reconstructs each optimisation level of MyModelFreeAgent from the final
checkpoint and measures raw-observation -> action latency at batch size 1:

  0. torchrl_actor   — full ProbabilisticActor via TensorDict (naive baseline)
  1. module_bypass   — call the raw nn.Sequential directly, sample in torch
  2. jit             — + torch.jit.script / optimize_for_inference
                       (Ant: trunk only, NormalParamExtractor is not scriptable)
  3. numpy_forward   — + numpy matmul forward with pre-allocated buffers
  4. fold_input      — + obs-norm folded into first layer, torch-free input prep
                       (= the shipped agent.get_action fast path)

Usage:
    python scripts/profile_inference_stages.py [--env all] [--n-obs 256]
        [--repeats 30] [--output results/inference_stages.json]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import gymnasium as gym

from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

import AAMAS_Comp  # noqa: F401 — registers the ExampleNS* environments
from AAMAS_Comp.agent import MyModelFreeAgent, _extract_np_layers

# Mirrors submission.get_agent model selection.
TARGETS = {
    "ant":        ("Ant-v5",        "models/ppo_ant/ppo_final.pt"),
    "cartpole":   ("CartPole-v1",   "models/ppo_cartpole/ppo_final.pt"),
    "frozenlake": ("FrozenLake-v1", "models/ppo_frozenlake/ppo_final_no_notify.pt"),
}

# NS environments used by the competition evaluator (for env.step latency).
NS_ENVS = {
    "ant":        "ExampleNSAnt-v0",
    "cartpole":   "ExampleNSCartPole-v0",
    "frozenlake": "ExampleNSFrozenLake-v0",
}

STAGES = ["torchrl_actor", "module_bypass", "jit", "numpy_forward", "fold_input"]


def collect_observations(env_id: str, n_obs: int, seed: int) -> list:
    """Roll out the plain gymnasium env with random actions to get raw obs."""
    env = gym.make(env_id, disable_env_checker=True)
    env.action_space.seed(seed)
    obs_list = []
    obs, _ = env.reset(seed=seed)
    for _ in range(n_obs):
        obs_list.append(obs)
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return obs_list


# ── Stage builders ────────────────────────────────────────────────────────────
# Each returns a callable raw_obs -> action replicating one historical
# implementation level. Stages 0-3 share the agent's _prepare_obs/_normalise.

def make_torchrl_actor(agent):
    actor, prepare, normalise = agent._actor, agent._prepare_obs, agent._normalise
    is_disc = agent._is_discrete

    def act(raw_obs):
        s = normalise(prepare(raw_obs))
        td = TensorDict({"observation": s.unsqueeze(0)}, batch_size=[1])
        with torch.no_grad():
            td = actor(td)
        a = td["action"].squeeze(0)
        if is_disc:
            return int(a.argmax().item()) if a.ndim else int(a.item())
        return a.numpy()

    return act


def _torch_sample(agent, net_out):
    """Sampling used by the bypass/JIT stages (same maths as _sample_action)."""
    if agent._is_discrete:
        logits = net_out.squeeze(0)
        noise = torch.empty_like(logits).exponential_().log_().neg_()
        return int((logits + noise).argmax().item())
    loc, scale = net_out
    loc, scale = loc.squeeze(0), scale.squeeze(0)
    raw = (loc + scale * torch.randn_like(loc)).tanh()
    if agent._action_low is not None:
        raw = agent._action_low + (raw + 1.0) * 0.5 * (agent._action_high - agent._action_low)
    return raw.numpy()


def make_module_bypass(agent):
    net, prepare, normalise = agent._raw_net, agent._prepare_obs, agent._normalise

    def act(raw_obs):
        s = normalise(prepare(raw_obs))
        with torch.no_grad():
            out = net(s.unsqueeze(0))
        return _torch_sample(agent, out)

    return act


def make_jit(agent):
    prepare, normalise = agent._prepare_obs, agent._normalise
    if agent._is_discrete:
        net = torch.jit.optimize_for_inference(torch.jit.script(agent._raw_net.eval()))

        def act(raw_obs):
            s = normalise(prepare(raw_obs))
            with torch.no_grad():
                out = net(s.unsqueeze(0))
            return _torch_sample(agent, out)
    else:
        # NormalParamExtractor uses *args and cannot be scripted — JIT trunk only.
        mlp = torch.jit.optimize_for_inference(torch.jit.script(agent._raw_net[0].eval()))
        npe = agent._raw_net[1]

        def act(raw_obs):
            s = normalise(prepare(raw_obs))
            with torch.no_grad():
                out = npe(mlp(s.unsqueeze(0)))
            return _torch_sample(agent, out)

    return act


def make_numpy_forward(agent):
    """Numpy matmul forward with separate (torch) normalisation, no folding."""
    layers = _extract_np_layers(agent._raw_net[0])  # unfolded weights
    bufs = [np.empty(b.shape[0], dtype=np.float32) for _, b, _ in layers]
    rng = np.random.default_rng()
    prepare, normalise = agent._prepare_obs, agent._normalise
    is_disc = agent._is_discrete
    if is_disc:
        gumbel_buf = np.empty(layers[-1][1].shape[0], dtype=np.float32)
    else:
        noise_buf = np.empty(layers[-1][1].shape[0] // 2, dtype=np.float32)

    def act(raw_obs):
        x = normalise(prepare(raw_obs)).numpy()
        for (W, b, act_fn), buf in zip(layers, bufs):
            np.dot(x, W, out=buf)
            buf += b
            if act_fn is not None:
                act_fn(buf, out=buf)
            x = buf
        if is_disc:
            rng.standard_exponential(out=gumbel_buf, dtype=np.float32)
            np.log(gumbel_buf, out=gumbel_buf)
            np.subtract(x, gumbel_buf, out=gumbel_buf)
            return int(gumbel_buf.argmax())
        act_dim = x.shape[0] // 2
        loc = x[:act_dim]
        scale = np.logaddexp(0.0, x[act_dim:] + agent._npe_bias) + agent._npe_min_val
        rng.standard_normal(out=noise_buf, dtype=np.float32)
        np.multiply(noise_buf, scale, out=noise_buf)
        np.add(noise_buf, loc, out=noise_buf)
        np.tanh(noise_buf, out=noise_buf)
        if agent._action_low is not None:
            lo = agent._action_low.numpy()
            hi = agent._action_high.numpy()
            return lo + (noise_buf + 1.0) * 0.5 * (hi - lo)
        return noise_buf

    return act


def make_fold_input(agent):
    """The shipped fast path: get_action with folded norm + np.copyto input."""
    assert agent._obs_buf is not None and not agent.online_learning
    return agent.get_action


STAGE_BUILDERS = {
    "torchrl_actor": make_torchrl_actor,
    "module_bypass": make_module_bypass,
    "jit":           make_jit,
    "numpy_forward": make_numpy_forward,
    "fold_input":    make_fold_input,
}


# ── Benchmark harness ─────────────────────────────────────────────────────────

def benchmark(fn, obs_list, repeats: int) -> dict:
    """Mean per-call latency, timed in whole-list chunks to amortise timer cost."""
    for obs in obs_list:  # warmup (JIT profiling runs, allocator, caches)
        fn(obs)
    chunk_means = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for obs in obs_list:
            fn(obs)
        chunk_means.append((time.perf_counter() - t0) / len(obs_list))
    arr = np.asarray(chunk_means) * 1e6  # µs
    return {
        "mean_us": float(arr.mean()),
        "std_us": float(arr.std()),
        "min_us": float(arr.min()),
    }


def measure_env_step(name: str, agent, n_steps: int, seed: int) -> dict:
    """Mean env.step latency in the NS env used by the competition evaluator.

    The agent drives the rollout (its final fast path) but only env.step is
    timed. notify-none matches the evaluator's default configuration.
    """
    env = gym.make(
        NS_ENVS[name],
        change_notification=False,
        delta_change_notification=False,
        disable_env_checker=True,
        order_enforce=False,
    )
    obs, _ = env.reset(seed=seed)
    times = []
    for _ in range(n_steps):
        action = agent.get_action(obs)
        t0 = time.perf_counter()
        obs, _, term, trunc, _ = env.step(action)
        times.append(time.perf_counter() - t0)
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    arr = np.asarray(times) * 1e6  # µs
    return {
        "mean_us": float(arr.mean()),
        "std_us": float(arr.std()),
        "p50_us": float(np.percentile(arr, 50)),
        "p95_us": float(np.percentile(arr, 95)),
        "n_steps": n_steps,
    }


def profile_env(name: str, n_obs: int, repeats: int, env_steps: int, seed: int) -> dict:
    env_id, model_path = TARGETS[name]
    obs_list = collect_observations(env_id, n_obs, seed)
    agent = MyModelFreeAgent(
        model_path, env_id=env_id, device="cpu",
        online_learning=False, use_ewc=False,
    )
    results = {}
    with set_exploration_type(ExplorationType.RANDOM):
        for stage in STAGES:
            fn = STAGE_BUILDERS[stage](agent)
            results[stage] = benchmark(fn, obs_list, repeats)
            print(f"  {name:<10} {stage:<14} mean={results[stage]['mean_us']:8.2f}µs"
                  f"  std={results[stage]['std_us']:.2f}µs")

    env_step = measure_env_step(name, agent, env_steps, seed)
    results["env_step"] = env_step
    print(f"  {name:<10} {'env_step':<14} mean={env_step['mean_us']:8.2f}µs"
          f"  p95={env_step['p95_us']:.2f}µs")

    base = results["torchrl_actor"]["mean_us"]
    for stage in STAGES:
        mean = results[stage]["mean_us"]
        results[stage]["speedup_vs_baseline"] = base / mean
        results[stage]["share_of_step"] = mean / (mean + env_step["mean_us"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=[*TARGETS, "all"], default="all")
    parser.add_argument("--n-obs", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--env-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/inference_stages.json")
    args = parser.parse_args()

    names = list(TARGETS) if args.env == "all" else [args.env]
    out = {
        "metadata": {
            "n_obs": args.n_obs,
            "repeats": args.repeats,
            "env_steps": args.env_steps,
            "seed": args.seed,
            "torch_num_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "stages": STAGES,
        },
        "results": {},
    }
    for name in names:
        print(f"\nProfiling {name} ({TARGETS[name][0]}) ...")
        out["results"][name] = profile_env(name, args.n_obs, args.repeats,
                                           args.env_steps, args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
