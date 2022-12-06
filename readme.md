# Mujoco Parallel Simulation
This repository restructures **_Deepmind's own implementation_** of parallel rollouts for Mujoco.
It adds the option to cache the trajectory buffers and the thread local MjModels since
in most cases these don't change.

- For use case take a look at the `tests/rollout_tests`.

__Example results__
```
Sequential rollout:      0.879
Parallel rollout:        0.087
Cached Parallel Rollout: 0.085
```

__To initialise__
```
$ pipenv shell
$ pipenv update
$ python3 rollout_tests.py
```

