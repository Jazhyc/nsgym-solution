#!/usr/bin/env bash
set -euo pipefail

run_trial() {
  local hidden_sizes="$1"
  local wandb_name="$2"

  echo "Running Ant with hidden_sizes=${hidden_sizes}"
  python scripts/train.py --config-name config_ant \
    "agent.hidden_sizes=${hidden_sizes}" \
    wandb.group=ant_hidden_size_sweep \
    "wandb.name=${wandb_name}"
}

run_trial '[256]' ant_h256_20m
run_trial '[512]' ant_h512_20m