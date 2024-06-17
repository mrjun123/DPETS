from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import torch


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    
    MODEL_IN, MODEL_OUT = 24, 18
    
    def __init__(self):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def _step(self, action):
        self.prev_qpos = np.copy(self.model.data.qpos.flat)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = ob[0] - 0.0 * np.square(ob[2])
        reward = reward_run + reward_ctrl
        done = False
        
        return ob, reward, done, {}

    def get_x(self):
        return self.model.data.qpos.flat[0]
    
    def _get_obs(self):
        return np.concatenate([
            (self.model.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
            self.model.data.qpos.flat[1:],
            self.model.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(loc=0, scale=0.001, size=self.model.nq)
        qvel = self.init_qvel + np.random.normal(loc=0, scale=0.001, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.model.data.qpos.flat)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55
        
    @staticmethod
    def obs_cost_fn_cost(obs):
        return -obs[..., 0]

    @staticmethod
    def ac_cost_fn_cost(acs):
        if isinstance(acs, np.ndarray):
            return 0.1 * np.sum(np.square(acs), axis=-1)
        else:
            return 0.1 * torch.sum(torch.square(acs), dim=-1)
        
    @staticmethod
    def obs_preproc(obs):
        dim = obs.ndim
        if isinstance(obs, np.ndarray):
            return np.concatenate([obs[..., 1:2], np.sin(obs[..., 2:3]), np.cos(obs[..., 2:3]), obs[..., 3:]], axis=-1)
        else:
            return torch.cat([obs[..., 1:2], torch.sin(obs[..., 2:3]), torch.cos(obs[..., 2:3]), obs[..., 3:]], dim=-1)

    @staticmethod
    def obs_postproc(obs, pred):
        if isinstance(obs, np.ndarray):
            return np.concatenate([pred[..., :1], obs[..., 1:] + pred[..., 1:]], axis=-1)
        else:
            return torch.cat([pred[..., :1], obs[..., 1:] + pred[..., 1:]], dim=-1)

    @staticmethod
    def targ_proc(obs, next_obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate([next_obs[..., :1], next_obs[..., 1:] - obs[..., 1:]], axis=-1)
        else:
            return torch.cat([next_obs[..., :1], next_obs[..., 1:] - obs[..., 1:]], dim=-1)
