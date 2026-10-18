"""Shared CartPole validity rules, matching native NS-Gym's constraints."""
import math


def valid_cartpole_parameter(name, value):
    if not math.isfinite(float(value)):
        return False
    if name in ('masspole', 'masscart', 'length'):
        return value > 0
    if name == 'gravity':
        return value >= 0
    return True
