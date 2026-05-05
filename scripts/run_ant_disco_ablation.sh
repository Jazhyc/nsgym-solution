#!/usr/bin/env bash
set -euo pipefail

python scripts/train.py --config-name config_ant \
  agent.use_disco=false \
  wandb.group=ant_disco_ablation \
  wandb.name=ant_disco_off

python scripts/train.py --config-name config_ant \
  agent.use_disco=true \
  wandb.group=ant_disco_ablation \
  wandb.name=ant_disco_on