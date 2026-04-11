# NS-Gym PPO Training

PPO agent for non-stationary environments (NS-Gym). Supports continuous control (Ant-v5), classic control (CartPole-v1), and discrete grid worlds (FrozenLake-v1). Configured via [Hydra](https://hydra.cc/) with Weights & Biases logging.

## Setup

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```

Copy `.env.example` to `.env` and set your WandB credentials (or disable WandB — see below).

## Quick Start

```bash
# Train on Ant-v5 (default)
python scripts/train.py

# Train on CartPole-v1
python scripts/train.py --config-name config_cartpole

# Train on FrozenLake-v1
python scripts/train.py --config-name config_frozenlake
```

## Config Structure

```
config/
    config.yaml            # Ant-v5 (default)
    config_cartpole.yaml   # CartPole-v1 — tuned frames, network, entropy
    config_frozenlake.yaml # FrozenLake-v1 — tuned frames, network, entropy
    agent/
        ppo.yaml           # PPO hyperparameters (lr, clip_epsilon, network arch, …)
    env/
        ant.yaml           # Ant-v5  (default)
        cartpole.yaml      # CartPole-v1 env settings
        frozenlake.yaml    # FrozenLake-v1 env settings
```

Each top-level config composes `agent/ppo.yaml` + `env/*.yaml` and then overrides training-loop settings with `_self_` taking precedence. Agent overrides (e.g. `entropy_coeff`, `hidden_sizes`) can live directly in the top-level config.

**`config/config.yaml`** — controls data collection and the training loop:

| Key | Default | Description |
|-----|---------|-------------|
| `collector.num_envs` | 24 | Total parallel environments |
| `collector.num_groups` | 2 | Async collector groups (overlap collection and training) |
| `collector.total_frames` | 5 000 000 | Total environment steps |
| `collector.frames_per_batch` | 2560 | Steps collected before each update |
| `collector.max_frames_per_traj` | 1000 | Episode truncation horizon |
| `training.num_epochs` | 5 | PPO inner epochs per batch |
| `training.sub_batch_size` | 256 | Mini-batch size |
| `training.target_kl` | 0.02 | KL early-stopping threshold (null to disable) |
| `training.eval_interval` | 128 | Evaluate every N collector iterations |
| `training.num_eval_episodes` | 8 | Parallel eval environments |
| `num_threads` | 4 | `torch.set_num_threads` (0 = PyTorch default) |

**`config/agent/ppo.yaml`** — PPO hyperparameters:

| Key | Default | Description |
|-----|---------|-------------|
| `lr` | 3e-4 | Initial learning rate |
| `lr_min` | 3e-5 | LR floor (prevents decay killing gradients) |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE λ |
| `clip_epsilon` | 0.2 | PPO clip ratio |
| `entropy_coeff` | 0.0 | Entropy bonus coefficient |
| `hidden_sizes` | [256, 256] | MLP hidden layer widths |
| `activation` | Tanh | Activation function (any `torch.nn` name) |
| `use_layer_norm` | true | LayerNorm after each hidden activation |
| `compile` | true | Enable `torch.compile` |

**`config/env/*.yaml`** — environment-specific settings:

| Key | Description |
|-----|-------------|
| `id` | Gymnasium environment ID |
| `frame_skip` | Action repeat (not applied by the script itself; informational) |
| `normalize_obs` | Normalize observations with running mean/std |
| `normalize_obs_init_steps` | Random steps used to bootstrap obs stats |
| `normalize_reward` | VecNormalize-style reward scaling (divide by running std) |

## Overriding Config via CLI

Hydra lets you override any config key on the command line. Use this to tune the pre-built configs without editing files:

```bash
# Extend FrozenLake training slightly and increase entropy bonus
python scripts/train.py --config-name config_frozenlake \
    collector.total_frames=150_000 \
    agent.entropy_coeff=0.02

# Ant — try a different learning rate and clip epsilon
python scripts/train.py agent.lr=1e-4 agent.clip_epsilon=0.1

# Run multiple seeds (Hydra multirun)
python scripts/train.py --config-name config_cartpole --multirun seed=1,2,3,4,5

# Disable WandB for a quick local run
python scripts/train.py --config-name config_frozenlake wandb.enabled=false
```

## Recommended Settings per Environment

| Environment | Config file | `total_frames` | Notes |
|------------|-------------|---------------|-------|
| Ant-v5 | `config.yaml` | 5 000 000 | 24 async envs, reward normalisation |
| CartPole-v1 | `config_cartpole.yaml` | 200 000 | Small [64,64] net; converges ~100k |
| FrozenLake-v1 | `config_frozenlake.yaml` | 100 000 | Entropy bonus 0.01; deteriorates if over-trained |

## Environment Details

### Ant-v5 (continuous control)
Continuous 27-D observation, continuous 8-D action (bounded). Actor uses a `TanhNormal` distribution. Observation and reward normalization are both enabled by default.

### CartPole-v1 (discrete action, continuous obs)
4-D continuous observation, discrete action (0 or 1). Actor uses a `Categorical` distribution. Observation normalization enabled; reward normalization off.

### FrozenLake-v1 (discrete action, discrete obs)
Integer observation (0–15) automatically one-hot encoded to a 16-D float vector by the training script. Discrete 4-action space. Actor uses a `Categorical` distribution. No observation or reward normalization.

## WandB Logging

```bash
# Disable WandB
python scripts/train.py wandb.enabled=false

# Change project/entity
python scripts/train.py wandb.project=my-project wandb.entity=my-team

# Run offline (sync later with `wandb sync`)
python scripts/train.py wandb.mode=offline
```

Logged metrics include `train/reward_mean`, `train/kl_approx`, `train/clip_fraction`, `train/epochs_done`, `train/policy_entropy`, and `eval/reward_sum`.

## Checkpoints

Ant training writes checkpoints to `checkpoints/` by default. When you have a run you trust, promote the final artifact into the tracked model slot with a manual copy, for example:

```bash
cp checkpoints/ppo_final.pt models/ppo_ant/ppo_final.pt
```

That tracked checkpoint includes the observation running statistics, so no separate normalizer file is needed.

Load a saved model:

```python
from AAMAS_Comp.agents.ppo import PPOAgent

agent = PPOAgent.load("models/ppo_ant/ppo_final.pt", device="cpu")
action = agent.get_action({"state": obs})
```
