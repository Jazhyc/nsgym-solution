## How to Use This?

Each evaluator now runs all three notification settings automatically:
- `notify-change`
- `notify-full`
- `notify-none`

So one command produces folders like:
- `results/ExampleNSFrozenLake-v0__notify-change`
- `results/ExampleNSFrozenLake-v0__notify-full`
- `results/ExampleNSFrozenLake-v0__notify-none`

You can change the experiments once per episode, with:
```bash
uv run python eval/eval_frozenlake.py --num-episodes 10 --start-seed 42 --change-step 1 --slip-probs 0.8 0.1 0.1
uv run python eval/eval_cartpole.py --num-episodes 10 --start-seed 42 --change-step 1 --masspole 0.2
uv run python eval/eval_ant.py --num-episodes 10 --start-seed 42 --change-step 500 --torso-mass 0.7
```

Or, change it multiple times within an episode with:
```bash
uv run python eval/eval_frozenlake.py --change-steps 1,20,40 --slip-probs-schedule "0.8,0.1,0.1"
uv run python eval/eval_cartpole.py --change-steps 50,100,150 --masspoles 0.2,0.4,0.1
uv run python eval/eval_ant.py --change-steps 250,500,750 --torso-masses 0.7,1.0,0.5
```

THE COMMAND I USED:
```bash
uv run python eval/eval_frozenlake.py --change-steps 10 --slip-probs-schedule "0.8,0.1,0.1"
uv run python eval/eval_cartpole.py --change-steps 50,100,150 --masspoles 0.2,5.9,0.0001
uv run python eval/eval_ant.py --change-steps 250,500,750 --torso-masses 0.0001,1.1,6.7
```