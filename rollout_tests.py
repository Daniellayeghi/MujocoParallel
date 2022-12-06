import time
import mujoco
import numpy as np
from absl.testing import parameterized
from par_rollout import ParallelRollouts

model_path = "cartpole.xml"


def get_state(data):
    return np.hstack((data.qpos, data.qvel, data.act))


def set_state(model, data, state):
    data.qpos = state[:model.nq]
    data.qvel = state[model.nq:model.nq + model.nv]
    data.act = state[model.nq + model.nv:model.nq + model.nv + model.na]


def step(model, data, state, **kwargs):
    if state is not None:
        set_state(model, data, state)
    for key, value in kwargs.items():
        if value is not None:
            setattr(data, key, np.reshape(value, getattr(data, key).shape))
    mujoco.mj_step(model, data)
    return get_state(data), data.sensordata


def single_rollout(model, data, initial_state, **kwargs):
    arg_nstep = set([a.shape[0] for a in kwargs.values()])
    assert len(arg_nstep) == 1  # nstep dimensions must match
    nstep = arg_nstep.pop()

    state = np.empty((nstep, model.nq + model.nv + model.na))
    sensordata = np.empty((nstep, model.nsensordata))

    mujoco.mj_resetData(model, data)
    for t in range(nstep):
        kwargs_t = {}
        for key, value in kwargs.items():
            kwargs_t[key] = value[0 if value.ndim == 1 else t]
        state[t], sensordata[t] = step(model, data,
                                       initial_state if t == 0 else None,
                                       **kwargs_t)
    return state, sensordata


def multi_rollout(model, data, initial_state, **kwargs):
    nstate = initial_state.shape[0]
    arg_nstep = set([a.shape[1] for a in kwargs.values()])
    assert len(arg_nstep) == 1  # nstep dimensions must match
    nstep = arg_nstep.pop()

    state = np.empty((nstate, nstep, model.nq + model.nv + model.na))
    sensordata = np.empty((nstate, nstep, model.nsensordata))
    for s in range(nstate):
        kwargs_s = {key: value[s] for key, value in kwargs.items()}
        state_s, sensordata_s = single_rollout(model, data, initial_state[s],
                                               **kwargs_s)
        state[s] = state_s
        sensordata[s] = sensordata_s
    return state.squeeze(), sensordata.squeeze()


class MuJoCoRolloutTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        np.random.seed(42)

    def test_rollout(self):
        model = mujoco.MjModel.from_xml_path(model_path)
        num_workers, nstate, nstep = 10, 100, 200
        initial_state = np.random.randn(nstate, model.nq + model.nv + model.na)
        state = np.zeros((nstate, nstep, model.nq + model.nv + model.na))
        sensordata = np.zeros((nstate, nstep, model.nsensordata))
        ctrl = np.random.randn(nstate, nstep, model.nu)

        data = mujoco.MjData(model)
        par_roll = ParallelRollouts(model_path, num_workers)
        now = time.time()
        res = par_roll(initial_state, state, ctrl, sensordata, use_cache=False)
        end1 = time.time()
        res_cached = par_roll(initial_state, state, ctrl, sensordata, use_cache=False)
        end2 = time.time()
        py_state, py_sensordata = multi_rollout(model, data, initial_state, ctrl=ctrl)
        end3 = time.time()
        print(
            f"Sequential rollout: {end3 - end2}, Parallel rollout: {end1 - now}, Cached Parallel Rollout {end2 - end1}"
        )
        np.testing.assert_array_equal(res, py_state)
        np.testing.assert_array_equal(res_cached, py_state)


if __name__ == '__main__':
  t = MuJoCoRolloutTest()
  t.test_rollout()
