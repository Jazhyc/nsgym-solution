from gymnasium.envs.registration import register

register(id="ExampleNSFrozenLake-v0",
         entry_point="AAMAS_Comp.examples.environments.nsFrozenlake:make_env",
         disable_env_checker=True,
         order_enforce=False
         )

register(id="ExampleNSCartPole-v0",
         entry_point="AAMAS_Comp.examples.environments.nsCartpole:make_env",
         disable_env_checker=True,
         order_enforce=False
         )

register(id="ExampleNSAnt-v0",
         entry_point="AAMAS_Comp.examples.environments.nsAnt:make_env",
         disable_env_checker=True,
         order_enforce=False
         )

del register