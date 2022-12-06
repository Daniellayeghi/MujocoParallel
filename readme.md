# Mujoco Parallel Simulation
This repository restructures Deepmind's own implementation of parallel rollouts for Mujoco.
It adds the option to cache the trajectory buffers and the thread local MjModels since
in most cases these don't change.

- For use case take a look at the `tests/rollout_tests`.

To initialise:
```
$ pipenv shell
$ pipenv update
$ python3 rollout_tests.py
```
