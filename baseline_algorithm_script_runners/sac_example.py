"""Example: Train SAC on stationary Ant-v5, evaluate on non-stationary NS-Gym Ant."""

from pathlib import Path
from stable_baselines3 import SAC
from AAMAS_Comp.examples.agents import AAMASCompBaselineSAC
from AAMAS_Comp.evaluation import run_complete_evaluation
import gymnasium as gym


MODELS_DIR = Path("models")


def train(total_timesteps=1_000_000, save_dir=MODELS_DIR, name_prefix="sac_ant"):
    """Train SAC on the stationary Ant-v5 environment.

    Args:
        total_timesteps (int): Total training timesteps.
        save_dir (Path): Directory to save the trained model.
        name_prefix (str): Name prefix for saved files.
    """
    if save_dir is None:
        save_path = MODELS_DIR / name_prefix
    else:
        save_path = save_dir / name_prefix

    env = gym.make("Ant-v5")

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device="auto"
    )
    model.learn(total_timesteps=total_timesteps)

    save_path.mkdir(parents=True, exist_ok=True)
    model.save(save_path / name_prefix)
    print(f"Model saved to {save_path}")

    env.close()
    return model, save_path


def evaluate(model, model_path=None, num_episodes=10, start_seed=42):
    """Evaluate a trained SAC model on the non-stationary Ant environment.

    Args:
        model: Trained SAC model instance.
        model_path (Path): Path to saved model (optional, loads from disk).
        num_episodes (int): Number of evaluation episodes.
        start_seed (int): Starting seed.
    """
    if model_path is not None:
        model = SAC.load(model_path)

    CHANGE_NOTIFICATION = True
    DELTA_CHANGE_NOTIFICATION = True

    ns_env = gym.make(
        "ExampleNSAnt-v0",
        change_notification=CHANGE_NOTIFICATION,
        delta_change_notification=DELTA_CHANGE_NOTIFICATION,
        disable_env_checker=True,
        order_enforce=False,
    )

    agent = AAMASCompBaselineSAC(model=model)

    run_complete_evaluation(
        env=ns_env,
        agent=agent,
        start_seed=start_seed,
        num_episodes=num_episodes,
        name_prefix="SAC_Ant",
    )

    ns_env.close()


def main():
    model, save_path = train()
    evaluate(model, model_path=save_path / "sac_ant.zip")


if __name__ == "__main__":
    main()
