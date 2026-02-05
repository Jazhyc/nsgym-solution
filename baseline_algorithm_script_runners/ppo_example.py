"""Example: Train PPO on stationary Ant-v5, evaluate on non-stationary NS-Gym Ant."""

from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from AAMAS_Comp.examples.agents import AAMASCompBaselinePPO
from AAMAS_Comp.evaluation import run_complete_evaluation
import gymnasium as gym


MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


def train(total_timesteps=1_000_000, save_dir=MODELS_DIR, name_prefix="ppo_ant"):
    """Train PPO on the stationary Ant-v5 environment.

    Args:
        total_timesteps (int): Total training timesteps.
        save_path (Path): Where to save the trained model. Defaults to models/ppo_ant.
    """
    if save_dir is None:
        save_path = MODELS_DIR / name_prefix 
    else:
        save_path = save_dir / name_prefix

    # Tuned hyperparameters from RL Zoo3 for Ant
    vec_env = DummyVecEnv([lambda: gym.make("Ant-v5")])
    env = VecNormalize(vec_env)

    model = PPO(
        "MlpPolicy",
        env,
        batch_size=32,
        n_steps=512,
        gamma=0.98,
        learning_rate=1.90609e-05,
        ent_coef=4.9646e-07,
        clip_range=0.1,
        n_epochs=10,
        gae_lambda=0.8,
        max_grad_norm=0.6,
        vf_coef=0.677239,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps)

    save_path.mkdir(parents=True, exist_ok=True)
    model.save(save_path / name_prefix)

    vec_normalize_file = name_prefix + "_vecnormalize.pkl"
    env.save(save_path / vec_normalize_file)
    print(f"Model saved to {save_path}")

    env.close()
    return model, save_path


def evaluate(model, model_path=None, vec_norm_path=None, num_episodes=10, start_seed=42):
    """Evaluate a trained PPO model on the non-stationary Ant environment.

    Args:
        model: Trained PPO model instance.
        save_path (Path): Path used during training (to find VecNormalize stats).
        num_episodes (int): Number of evaluation episodes.
        start_seed (int): Starting seed.
    """
    # Load VecNormalize stats so observations are normalized before prediction
    vec_normalize = None
    if vec_norm_path is not None:
        dummy_env = DummyVecEnv([lambda: gym.make("Ant-v5")])
        vec_normalize = VecNormalize.load(vec_norm_path, dummy_env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False


    if model_path is not None:
        model = model.load(model_path)


    CHANGE_NOTIFICATION = True
    DELTA_CHANGE_NOTIFICATION = True

    ns_env = gym.make(
        "ExampleNSAnt-v0",
        change_notification=CHANGE_NOTIFICATION,
        delta_change_notification=DELTA_CHANGE_NOTIFICATION,
        disable_env_checker=True,
        order_enforce=False,
    )

    agent = AAMASCompBaselinePPO(model=model, vec_normalize=vec_normalize)

    run_complete_evaluation(
        env=ns_env,
        agent=agent,
        start_seed=start_seed,
        num_episodes=num_episodes,
        name_prefix="PPO_Ant",
    )

    ns_env.close()


def main():
    model, vec_norm_path = train()
    evaluate(model, model_path=Path("models/ppo_ant/ppo_ant.zip"), vec_norm_path=Path("models/ppo_ant/ppo_ant_vecnormalize.pkl"))


if __name__ == "__main__":
    main()
