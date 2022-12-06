# Mujoco Parallel Simulation
This repository restructures **_Deepmind's own implementation_** (as example bindings) of parallel rollouts for Mujoco.
It adds the option to cache the trajectory buffers and the thread local MjModels since
in most cases these don't change.

- For use case take a look at the `tests/rollout_tests`.

__Example results__
```
Sequential rollout:      0.879s
Parallel rollout:        0.087s
Cached Parallel Rollout: 0.085s
```

__To initialise__
```
$ pipenv shell
$ pipenv update
$ python3 rollout_tests.py
```

