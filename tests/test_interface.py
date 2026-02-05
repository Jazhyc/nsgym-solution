"""Tests for agent interface conformance (base_agent.py, agent.py, submission.py)."""

import inspect
import pytest
import numpy as np
from ns_gym.base import Agent
from AAMAS_Comp.base_agent import ModelBasedAgent, ModelFreeAgent, SB3Agent
from AAMAS_Comp.agent import MyModelBasedAgent, MyModelFreeAgent


# ── base_agent.py: inheritance hierarchy ─────────────────────────────────────

class TestBaseAgentHierarchy:

    def test_model_based_is_agent_subclass(self):
        assert issubclass(ModelBasedAgent, Agent)

    def test_model_free_is_agent_subclass(self):
        assert issubclass(ModelFreeAgent, Agent)

    def test_sb3_is_model_free_subclass(self):
        assert issubclass(SB3Agent, ModelFreeAgent)

    def test_model_based_get_action_raises(self):
        agent = ModelBasedAgent()
        with pytest.raises(NotImplementedError):
            agent.get_action({}, None)

    def test_model_free_get_action_raises(self):
        agent = ModelFreeAgent()
        with pytest.raises(NotImplementedError):
            agent.get_action({})


# ── base_agent.py: method signatures ─────────────────────────────────────────

class TestBaseAgentSignatures:

    def test_model_based_get_action_params(self):
        params = list(inspect.signature(ModelBasedAgent.get_action).parameters)
        assert "obs" in params
        assert "planning_env" in params

    def test_model_free_get_action_params(self):
        params = list(inspect.signature(ModelFreeAgent.get_action).parameters)
        assert "obs" in params
        assert "planning_env" not in params

    def test_model_based_has_validate_and_get_action(self):
        assert hasattr(ModelBasedAgent, "validate_and_get_action")
        assert callable(ModelBasedAgent.validate_and_get_action)

    def test_model_free_has_validate_and_get_action(self):
        assert hasattr(ModelFreeAgent, "validate_and_get_action")
        assert callable(ModelFreeAgent.validate_and_get_action)

    def test_model_based_has_act(self):
        assert hasattr(ModelBasedAgent, "act")

    def test_model_free_has_act(self):
        assert hasattr(ModelFreeAgent, "act")


# ── base_agent.py: validate_and_get_action with dummy agents ─────────────────

class TestValidateAndGetAction:

    def test_model_based_returns_action_and_time(
        self, dummy_model_based_agent, ns_frozenlake_env
    ):
        obs, _ = ns_frozenlake_env.reset(seed=42)
        planning_env = ns_frozenlake_env.get_planning_env()
        result = dummy_model_based_agent.validate_and_get_action(obs, planning_env)

        assert isinstance(result, tuple)
        assert len(result) == 2
        action, dt = result
        assert isinstance(action, (np.ndarray, int, np.integer))
        assert isinstance(dt, float)
        assert dt >= 0
        assert planning_env.action_space.contains(action)

    def test_model_free_returns_action_and_time(
        self, dummy_model_free_agent_factory, ns_cartpole_env
    ):
        agent = dummy_model_free_agent_factory(ns_cartpole_env.action_space)
        obs, _ = ns_cartpole_env.reset(seed=42)
        result = agent.validate_and_get_action(obs, ns_cartpole_env.action_space)

        assert isinstance(result, tuple)
        assert len(result) == 2
        action, dt = result
        assert isinstance(dt, float)
        assert dt >= 0
        assert ns_cartpole_env.action_space.contains(action)

    def test_model_based_action_in_space(
        self, dummy_model_based_agent, ns_frozenlake_env
    ):
        obs, _ = ns_frozenlake_env.reset(seed=0)
        planning_env = ns_frozenlake_env.get_planning_env()
        for _ in range(5):
            action = dummy_model_based_agent.get_action(obs, planning_env)
            assert planning_env.action_space.contains(action)

    def test_model_free_action_in_space(
        self, dummy_model_free_agent_factory, ns_cartpole_env
    ):
        agent = dummy_model_free_agent_factory(ns_cartpole_env.action_space)
        obs, _ = ns_cartpole_env.reset(seed=0)
        for _ in range(5):
            action = agent.get_action(obs)
            assert ns_cartpole_env.action_space.contains(action)


# ── agent.py: participant stubs ──────────────────────────────────────────────

class TestAgentStubs:

    def test_my_model_based_agent_is_model_based(self):
        assert issubclass(MyModelBasedAgent, ModelBasedAgent)

    def test_my_model_free_agent_is_model_free(self):
        assert issubclass(MyModelFreeAgent, ModelFreeAgent)

    def test_my_model_based_has_get_action(self):
        assert hasattr(MyModelBasedAgent, "get_action")

    def test_my_model_free_has_get_action(self):
        assert hasattr(MyModelFreeAgent, "get_action")


# ── submission.py: get_agent ─────────────────────────────────────────────────

class TestSubmission:

    def test_get_agent_is_callable(self):
        from submission import get_agent
        assert callable(get_agent)

    @pytest.mark.parametrize("env_id", ["FrozenLake-v1", "CartPole-v1", "Ant-v5"])
    def test_get_agent_returns_valid_agent_or_not_implemented(self, env_id):
        from submission import get_agent
        try:
            agent = get_agent(env_id)
        except NotImplementedError:
            pytest.skip(f"submission.py not yet implemented for {env_id}")
        assert isinstance(agent, (ModelBasedAgent, ModelFreeAgent)), (
            f"get_agent('{env_id}') must return a ModelBasedAgent or ModelFreeAgent, "
            f"got {type(agent)}"
        )

    def test_get_agent_unknown_env_raises(self):
        from submission import get_agent
        with pytest.raises((NotImplementedError, ValueError)):
            get_agent("NonExistentEnv-v99")
