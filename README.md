# AAMAS 2026 Competition: Evaluating Adaptive Decision Agents under Non-Stationarity

This repository contains evaluation code, example agents, and boilerplate to get started with the [NS-Gym](https://nsgym.io) framework for the [AAMAS 2026 Competition](https://nsgym.io/aamas2026_competition.html). 

## Summary 

### What Is The Competition?

This competition, co-located with AAMAS 2026, invites researchers and practitioners to develop and evaluate decision-making agents that can effectively adapt to non-stationary environments using the NS-Gym framework. Participants will design agents capable of detecting, adapting to, and recovering from dynamic environmental shifts. The solution method is open to any approach, including but not limited to reinforcement learning, online planning, meta-learning, and continuous learning. The competition aims to foster innovation in adaptive decision-making and provide insights into the challenges of non-stationary environments. Results will be presented at the AAMAS 2026 conference. 

The competition will focus on designing decision-making agents for three base environments: [FrozenLake](https://nsgym.io/env_pages/toytext/frozenlake.html), [CartPole](https://nsgym.io/env_pages/classic_control/cartpole.html), and [MuJoCo Ant](https://nsgym.io/env_pages/mujoco/antenv.html). While the problem class remains the same, the nature of non-stationarity will differ. We provide three pre-configured non-stationary versions of these environments. During evaluation, we will evaluate performance on these environments and a holdout set of other non-stationary configurations. Please see the [Environments](#environments) section of this README for details about pre-configured environments and how to create your own non-stationary MDPs.

### What Will You Implement?

1. [src/AAMAS_Comp/agent.py](src/AAMAS_Comp/agent.py): Implement your core agent logic here. This integrates the agent as part of the `AAMAS_Comp` package. See example agents in this directory [src/AAMAS_Comp/examples/agents](src/AAMAS_Comp/examples/agents).
2. [submission.py](submission.py): Fill in the `get_agent()` function in this file to configure your agent for each environment: load model weights, set hyperparameters, etc. See [example_submission.py](example_submission.py) for an example.
3. Save model weights in the [models](models) directory. 
4. If you have additional Python dependencies, add them to `pyproject.toml` using `uv add <package-name>`. If you have non-Python dependencies (system packages, etc.), add them to [eval.Dockerfile](docker/eval.Dockerfile).

### How Can You Evaluate Your Agent? 

Running the [evaluator.py](evaluator.py) script will evaluate your agent against all provided example environments. Feel free to edit the `ENVIRONMENTS` dictionary to add additional configured non-stationary environments. This script will generate a `results` directory that contains evaluation results. Please see the [Environments](#environments) section of this README for details about pre-configured environments and how to create your own non-stationary MDPs.

```bash
python evaluator.py
```

To verify your submission matches the competition evaluation environment, build and run the Docker container:

```bash
docker compose build              # builds base and evaluation images
docker compose run test-submission  # runs evaluation inside the container
```

This builds a `linux/amd64` container matching the competition evaluation environment: 32 GB RAM, single NVIDIA GPU. You only need to re-run `docker compose build` after changing dependencies in `pyproject.toml` or `docker/eval.Dockerfile`. Code changes to `submission.py`, `src/`, and `evaluator.py` are mounted and picked up automatically.

## Repo Structure

```
evaluator.py                            # Runs submitted agent against all environments
submission.py                           # YOUR entrypoint -- wire up your agent here
src/AAMAS_Comp/
    __init__.py                         # Environment registration
    base_agent.py                       # ModelBasedAgent, ModelFreeAgent, SB3Agent
    agent.py                            # YOUR agent implementation goes here
    evaluation/
        evaluate_agent.py               # Episode runner and evaluation harness
        utils.py                        # Metric utilities
    examples/
        agents/                         # Baseline agent wrappers
            mcts_example.py             # MCTS (model-based)
            ppo_example.py              # PPO (model-free, SB3)
            sac_example.py              # SAC (model-free, SB3)
        environments/                   # Pre-configured NS environments
            nsFrozenlake.py
            nsCartpole.py
            nsAnt.py
baseline_algorithm_script_runners/      # Standalone train/eval scripts
    mcts_example.py
    ppo_example.py
    sac_example.py
docker/
    base.Dockerfile                     # Base image with dependencies -- DO NOT MODIFY
    eval.Dockerfile                     # Evaluation image -- add custom deps here
docker-compose.yml                      # Dev and test-submission services
```


## Development Environment Set Up

1. Click **"Use this template"** at the top of this page to create your own repository (private or public) called `ns-gym-comp-submission`.

2. Clone your new repository and add this template as an upstream remote so you can pull future updates (new environments, examples, evaluation changes, etc.):

```bash
git clone https://github.com/{your-username}/ns-gym-comp-submission.git
cd ns-gym-comp-submission
git remote add upstream https://github.com/scope-lab-vu/ns-gym-comp-template.git
```

3. Add the competition organizers -- [nkepling](https://github.com/nkepling) and [ayanmukhopadhyay](https://github.com/ayanmukhopadhyay) -- as **collaborators** on your repository so we can clone, run, and evaluate your submission.

4. When ready to submit, [open an issue](https://github.com/scope-lab-vu/ns-gym-comp-template/issues/new) on this template repository with a link to your submission repo.

To pull template updates at any time:

```bash
git fetch upstream
git merge upstream/main --allow-unrelated-histories
```


### Virtual Environment

After cloning your repository, create a virtual environment and install the `AAMAS_Comp` package. Install [uv](https://docs.astral.sh/uv/getting-started/installation), then:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```

This installs NS-Gym, Stable-Baselines3, PyTorch, Gymnasium, MuJoCo, and other dependencies. Verify with `uv pip list`. To install additional dependencies, add them with `uv add <package-name>`.

### Docker

The Docker setup uses two images:

- **`docker/base.Dockerfile`** -- base image with Python, uv, and project dependencies. **Do not modify this file.**
- **`docker/eval.Dockerfile`** -- evaluation image that builds on the base. Edit this file to install additional non-Python dependencies (system packages, etc.) your agent needs.

Build both images:

```bash
docker compose build
```

All submitted agents will be evaluated using this Docker image on a `linux/amd64` machine with 32 GB RAM and a single NVIDIA GPU. GPU access is available inside the container — agents can use CUDA via PyTorch. Ensure your agent runs correctly within this environment.



## Developing Your Agent

Implement your agent in [src/AAMAS_Comp/agent.py](src/AAMAS_Comp/agent.py). Your agent must subclass either `ModelBasedAgent` or `ModelFreeAgent` from `base_agent.py`. Import all dependencies into this file.

**Do not modify `base_agent.py`** -- the competition evaluator uses its own copy.

### Agent Types

**ModelBasedAgent** -- receives a planning environment for lookahead search:

```python
class ModelBasedAgent(Agent):
    def get_action(self, obs: Dict, planning_env: gym.Env):
        # Use planning_env for simulation/search
        ...
```

**ModelFreeAgent** -- receives only the observation:

```python
class ModelFreeAgent(Agent):
    def get_action(self, obs: Dict):
        # Select action from observation alone
        ...
```

Both agent types are timed via `validate_and_get_action()` during evaluation. Actions are validated against the action space.

Add any new dependencies to `pyproject.toml`:

```bash
uv add <package-name>
```

## Environments

Three non-stationary environments are registered and ready to use. Each wraps a standard Gymnasium environment with NS-Gym schedulers and update functions that modify environment parameters over time. Environment source code is in [src/AAMAS_Comp/examples/environments/](src/AAMAS_Comp/examples/environments/).

All environments accept `change_notification` and `delta_change_notification` kwargs to control whether the agent is informed of parameter changes.

```python
import gymnasium as gym
import AAMAS_Comp  # triggers environment registration

env = gym.make("ExampleNSFrozenLake-v0", change_notification=True, delta_change_notification=True)
```

### ExampleNSFrozenLake-v0

Wraps `FrozenLake-v1` (discrete, 16 states, 4 actions). Transition probabilities `P` are decremented by 0.025 each step via `DistributionDecrementUpdate` on a `ContinuousScheduler`, making the surface progressively more slippery.

### ExampleNSCartPole-v0

Wraps `CartPole-v1` (continuous state, 2 discrete actions). Two parameters change simultaneously:
- **masspole**: increases by 0.1 each step (`IncrementUpdate`, `ContinuousScheduler`)
- **gravity**: random walk every 3 steps (`RandomWalk`, `PeriodicScheduler`)

### ExampleNSAnt-v0

Wraps `Ant-v5` (continuous state/action, MuJoCo). The `torso_mass` decays exponentially with rate 0.9 (`ExponentialDecay`, `PeriodicScheduler` with period 500, active from step 100 to 500), making the ant progressively lighter.

### Competition Evaluation Environments

Final evaluation will use the same three base environments (FrozenLake, CartPole, Ant), but the nature of non-stationarity will differ from the pre-configured examples above. The specific schedulers, update functions, tunable parameters, and their configurations will be different and are not disclosed in advance.

Participants are encouraged to construct and experiment with different non-stationarity conditions beyond the provided examples to build agents that generalize across varying forms of environmental change. See the [NS-Gym documentation](https://nsgym.io) for available schedulers, update functions, and tunable parameters.

### Making Your Own Custom Environment

You can create custom non-stationary environments by combining three components:

1. **Base environment** -- one of the competition environments: `CartPole-v1`, `FrozenLake-v1`, or `Ant-v5`.
2. **Schedulers** -- control *when* parameter changes occur. For example, `ContinuousScheduler` applies changes every step, while `PeriodicScheduler(period=n)` applies changes every `n` steps.
3. **Update functions** -- define *how* parameters change. Each update function is paired with a scheduler. For example, `IncrementUpdate(scheduler, k=0.1)` adds a fixed increment each time the scheduler fires, and `RandomWalk(scheduler)` applies a random perturbation.

Pair each tunable parameter name (matching the environment's internal attributes) with an update function in a dictionary, then pass everything to the appropriate NS-Gym wrapper (`NSClassicControlWrapper`, `NSFrozenLakeWrapper`, `NSMujocoWrapper`, etc.). You can also enable `change_notification` to receive alerts when parameters change and `delta_change_notification` to receive the magnitude of each change. The NS-Gym documentation has tables of all tunable parameters for each environment (e.g., [CartPole](https://nsgym.io/env_pages/classic_control/cartpole.html)). See [src/AAMAS_Comp/examples/environments/](src/AAMAS_Comp/examples/environments/) for full examples.

The code below creates a CartPole environment where the cart mass oscillates every step and the pole mass follows a random walk every 5 steps:

```python
import gymnasium as gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import RandomWalk, OscillatingUpdate

def make_env(**kwargs):
    change_notification = kwargs.get("change_notification", False)
    delta_change_notification = kwargs.get("delta_change_notification", False)

    base_env = gym.make("CartPole-v1")

    scheduler_1 = ContinuousScheduler()
    scheduler_2 = PeriodicScheduler(period=5)

    update_function_1 = OscillatingUpdate(scheduler_1)
    update_function_2 = RandomWalk(scheduler_2)

    tunable_params = {"masscart": update_function_1, "masspole": update_function_2}

    ns_env = NSClassicControlWrapper(base_env,
                                     tunable_params,
                                     change_notification=change_notification,
                                     delta_change_notification=delta_change_notification)
    return ns_env
```

To reuse this environment elsewhere, register it with Gymnasium's registration API. Add a `register()` call in [src/AAMAS_Comp/\_\_init\_\_.py](src/AAMAS_Comp/__init__.py) pointing to your `make_env` function:

```python
register(id="MyCustomNSCartPole-v0",
         entry_point="AAMAS_Comp.examples.environments.my_custom_env:make_env",
         disable_env_checker=True,
         order_enforce=False)
```

You can then load it anywhere with `gym.make("MyCustomNSCartPole-v0")` and add the env ID to the `ENVIRONMENTS` dictionary in [evaluator.py](evaluator.py) to evaluate your agent on it.

#### Observation Format

All NS-Gym environments return observations as a dictionary:

```python
obs:  {'state': array([-0.03006991,  0.19717823,  0.02711801, -0.3215324 ], dtype=float32),
        'env_change': {'masscart': 1, 'masspole': 0},
        'delta_change': {'masscart': 0.112, 'masspole': 0.0},
        'relative_time': 1}
```

- `state`: the standard Gymnasium observation of the environment.
- `env_change`: a dictionary indicating whether each parameter has changed (1 indicates a change, 0 indicates no change). Only available if `change_notification=True`.
- `delta_change`: a dictionary indicating the magnitude of change for each parameter. Only available if `delta_change_notification=True`. Defaults to zero if not set.
- `relative_time`: current time step of the environment.

#### Planning Environment

NS-Gym provides an interface to grab a **stationary** version of the environment for planning by calling `ns_env.get_planning_env()`. This returns a copy of the environment in accordance with the notification settings. If `delta_change_notification` is set to `True`, the most up-to-date version of the environment is returned (though no future evolutions). Otherwise the first known version of the MDP is returned.

#### Execution Trace Example

To better understand the notification system, `get_planning_env()`, and the evolution of MDPs, consider the following execution trace. Suppose we have an initial MDP $\mathcal{M}_0$ whose transition function is parametrized by parameter $\theta_0$, and NS-Gym is configured to update $\theta$ every two decision epochs.

| | $t_1$ | $t_2$ | $t_3$ | $t_4$ | $t_5$ | $t_6$ | $t_7$ | $t_8$ | $t_9$ | $t_{10}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| **MDP** | $\mathcal{M}_0$ | $\mathcal{M}_0$ | $\mathcal{M}_1$ | $\mathcal{M}_1$ | $\mathcal{M}_2$ | $\mathcal{M}_2$ | $\mathcal{M}_3$ | $\mathcal{M}_3$ | $\mathcal{M}_4$ | $\mathcal{M}_4$ |
| $\theta$ | $\theta_0$ | $\theta_0$ | $\theta_1$ | $\theta_1$ | $\theta_2$ | $\theta_2$ | $\theta_3$ | $\theta_3$ | $\theta_4$ | $\theta_4$ |

At initialization ($t_0$), the agent always knows $\mathcal{M}_0$ and $\theta_0$. Consider the transition from $t_6$ to $t_7$.

If `change_notification == True` and `delta_change_notification == False`, the agent is notified that we have transitioned from $\mathcal{M}_2$ to $\mathcal{M}_3$ but **does not know** $\theta_3$.

If `change_notification == True` and `delta_change_notification == True`, the agent is notified that we have transitioned from $\mathcal{M}_2$ to $\mathcal{M}_3$ and **does know** $\theta_3$.

If `change_notification == False` (and by default `delta_change_notification == False`), the agent is not notified of any changes and only has information about $\mathcal{M}_0$ and $\theta_0$.

Suppose after transitioning from $t_6$ to $t_7$ we call `planning_env = ns_env.get_planning_env()`.

If `change_notification == True` and `delta_change_notification == False`, `planning_env` will be a stationary copy of $\mathcal{M}_0$ since we do not know $\theta_3$.

If `change_notification == True` and `delta_change_notification == True`, `planning_env` will be a stationary copy of $\mathcal{M}_3$ because we do know $\theta_3$.

If `change_notification == False` and `delta_change_notification == False`, `planning_env` will be a stationary copy of $\mathcal{M}_0$.


## Examples

Standalone scripts in the [baseline_algorithm_script_runners/](baseline_algorithm_script_runners/) directory demonstrate training and evaluation of baseline algorithms. Corresponding agent wrappers are in [src/AAMAS_Comp/examples/agents/](src/AAMAS_Comp/examples/agents/).

### MCTS (`examples/mcts_example.py`)

Evaluates the NS-Gym MCTS implementation (with chance nodes) on `ExampleNSFrozenLake-v0`. This is a **model-based** agent that uses the planning environment for tree search. Configured with rollout depth 50, 100 iterations, UCT constant 1.4, and discount 0.99.

```bash
python examples/mcts_example.py
```

### PPO (`examples/ppo_example.py`)

Trains Stable-Baselines3 PPO on stationary `Ant-v5` with RL Zoo3 tuned hyperparameters and `VecNormalize`, then evaluates on `ExampleNSAnt-v0`. This is a **model-free** agent. VecNormalize statistics are saved alongside the model and loaded during evaluation to normalize observations.

```bash
python examples/ppo_example.py
```

### SAC (`examples/sac_example.py`)

Trains Stable-Baselines3 SAC on stationary `Ant-v5` with default hyperparameters, then evaluates on `ExampleNSAnt-v0`. This is a **model-free** agent. SAC does not require `VecNormalize`, making the pipeline simpler than PPO.

```bash
python examples/sac_example.py
```

## Evaluation

The evaluation harness is in [src/AAMAS_Comp/evaluation/evaluate_agent.py](src/AAMAS_Comp/evaluation/evaluate_agent.py). The competition uses the same harness to evaluate all submissions.

```python
from AAMAS_Comp.evaluation import run_complete_evaluation

run_complete_evaluation(
    env=ns_env,
    agent=agent,
    start_seed=42,
    num_episodes=10,
    name_prefix="MyAgent",
)
```

This runs `num_episodes` episodes with deterministic sequential seeding (`start_seed`, `start_seed + 1`, ...) and saves to `results/{name_prefix}/`:
- `{name_prefix}.zip` -- compressed per-step metrics (observations, actions, rewards, decision times, environment changes)
- `metadata.json` -- seed range, episode count, total wall time
- `summary.json` -- per-episode and aggregate statistics (mean reward, decision time, number of transition function changes)

### Docker Evaluation

Edit [submission.py](submission.py) to wire up your agent. The evaluator calls `get_agent(env_id)` once per base environment, passing the environment ID (`"FrozenLake-v1"`, `"CartPole-v1"`, or `"Ant-v5"`). Use this to load environment-specific model weights, hyperparameters, or agent classes:

```python
def get_agent(env_id: str):
    if env_id == "Ant-v5":
        model = PPO.load("models/ppo_ant/ppo_ant.zip")
        return MyModelFreeAgent(model=model)
    elif env_id == "FrozenLake-v1":
        return MyModelBasedAgent(d=50, m=100)
    elif env_id == "CartPole-v1":
        return MyModelFreeAgent()
```

Then test locally:

```bash
docker compose build              # first time, or after changing dependencies
docker compose run test-submission
```

This runs [evaluator.py](evaluator.py) inside the container against all three competition environments. The `test-submission` service mounts your `submission.py`, `src/`, and `models/` so code and weight changes are picked up without rebuilding. You only need to re-run `docker compose build` after modifying `pyproject.toml` or `docker/eval.Dockerfile`.

## Submission

1. Ensure the competition organizers ([nkepling](https://github.com/nkepling), [ayanmukhopadhyay](https://github.com/ayanmukhopadhyay)) are collaborators on your repository.
2. Verify your submission runs successfully with `docker compose run test-submission`.
3. [Open an issue](https://github.com/scope-lab-vu/ns-gym-comp-template/issues/new) on this template repository with a link to your submission repo.
