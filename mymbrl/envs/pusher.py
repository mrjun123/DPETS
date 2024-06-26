from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box
import torch
import gym

class PusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    MODEL_IN, MODEL_OUT = 27, 20
    
    def __init__(self):
        # observation_space = Box(
        #     low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64
        # )
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/pusher.xml' % dir_path, 4)
        utils.EzPickle.__init__(self)
        self.reset_model()

    def _step(self, a):
        obj_pos = self.get_body_com("object"),
        vec_1 = obj_pos - self.get_body_com("tips_arm")
        vec_2 = obj_pos - self.get_body_com("goal")

        reward_near = -np.sum(np.abs(vec_1))
        reward_dist = -np.sum(np.abs(vec_2))
        reward_ctrl = -np.square(a).sum()
        reward = 1.25 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, {'reward': reward}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        self.cylinder_pos = np.array([-0.25, 0.15]) + np.random.normal(0, 0.025, [2])

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        self.ac_goal_pos = self.get_body_com("goal")
        # self.ac_goal_pos_tensor = torch.tensor(self.ac_goal_pos, device="cpu")

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
        ])
        
    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred
    
    @staticmethod
    def obs_preproc(obs):
        return obs

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    def obs_cost_fn_cost(self, obs):
        ndim = obs.ndim
        to_w, og_w = 0.5, 1.25
        if ndim == 2:
            tip_pos, obj_pos, goal_pos = obs[:, 14:17], obs[:, 17:20], self.ac_goal_pos
        elif ndim == 3:
            tip_pos, obj_pos, goal_pos = obs[:, :, 14:17], obs[:, :, 17:20], self.ac_goal_pos

        if isinstance(obs, np.ndarray):
            tip_obj_dist = np.sum(np.abs(tip_pos - obj_pos), axis=-1)
            obj_goal_dist = np.sum(np.abs(goal_pos - obj_pos), axis=-1)
            return to_w * tip_obj_dist + og_w * obj_goal_dist
        else:
            # goal_pos = self.ac_goal_pos_tensor
            goal_pos = torch.tensor(goal_pos, device=obs.device)
            tip_obj_dist = torch.sum(torch.abs(tip_pos - obj_pos), dim=-1)
            obj_goal_dist = torch.sum(torch.abs(goal_pos - obj_pos), dim=-1)
            return to_w * tip_obj_dist + og_w * obj_goal_dist
    
    @staticmethod
    def ac_cost_fn_cost(acs):
        if isinstance(acs, np.ndarray):
            return 0.1 * np.sum(np.square(acs), axis=1)
        else:
            return 0.1 * torch.sum(torch.square(acs), dim=1)