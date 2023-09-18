from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import torch


class CartpoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    PENDULUM_LENGTH = 0.6
    MODEL_IN, MODEL_OUT = 6, 4
    OBS_ADD_DIM = 1

    def __init__(self):
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/cartpole.xml' % dir_path, 2)

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        cost_lscale = CartpoleEnv.PENDULUM_LENGTH
        reward = np.exp(
            -np.sum(np.square(self._get_ee_pos(ob) - np.array([0.0, CartpoleEnv.PENDULUM_LENGTH]))) / (cost_lscale ** 2)
        )
        reward -= 0.01 * np.sum(np.square(a))

        done = False
        return ob, reward, done, {'reward': reward}

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(0, 0.1, np.shape(self.init_qpos))
        qvel = self.init_qvel + np.random.normal(0, 0.1, np.shape(self.init_qvel))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.model.data.qpos, self.model.data.qvel]).ravel()
    
    
    @staticmethod
    def _get_ee_pos(x):
        x0, theta = x[0], x[1]
        return np.array([
            x0 - CartpoleEnv.PENDULUM_LENGTH * np.sin(theta),
            -CartpoleEnv.PENDULUM_LENGTH * np.cos(theta)
        ])

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = v.model.stat.extent
        
    @staticmethod
    def obs_cost_fn_cost(obs):
        if isinstance(obs, np.ndarray):
            return -np.exp(-np.sum(
                np.square(CartpoleEnv._get_ee_pos_cost(obs) - np.array([0.0, 0.6])), axis=1
            ) / (0.6 ** 2))
        else:
            return -torch.exp(-torch.sum(
                torch.square(CartpoleEnv._get_ee_pos_cost(obs) - torch.tensor([0.0, 0.6], device=obs.device)), dim=1
            ) / (0.6 ** 2))

    @staticmethod
    def ac_cost_fn_cost(acs):
        if isinstance(acs, np.ndarray):
            return 0.01 * np.sum(np.square(acs), axis=1)
        else:
            return 0.01 * torch.sum(torch.square(acs), dim=1)
    
        
    @staticmethod
    def _get_ee_pos_cost(obs):
        x0, theta = obs[:, :1], obs[:, 1:2]
        if isinstance(obs, np.ndarray):
            return np.concatenate([x0 - 0.6 * np.sin(theta), -0.6 * np.cos(theta)], axis=1)
        else:
            return torch.cat([x0 - 0.6 * torch.sin(theta), -0.6 * torch.cos(theta)], dim=1)

    @staticmethod
    def obs_preproc(obs):
        if isinstance(obs, np.ndarray):
            dim = obs.ndim
            if dim == 1:
                return np.concatenate([np.sin(obs[1:2]), np.cos(obs[1:2]), obs[:1], obs[2:]], axis=-1)
            return np.concatenate([np.sin(obs[:, 1:2]), np.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:]], axis=1)
        else:
            dim = obs.ndim
            if dim == 3:
                return torch.cat((torch.sin(obs[:,:, 1:2]), torch.cos(obs[:,:, 1:2]), obs[:,:, :1], obs[:,:, 2:]), dim=2)
            return torch.cat((torch.sin(obs[:, 1:2]), torch.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:]), dim=1)

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs
