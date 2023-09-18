import mymbrl.optimizers as optimizers
import numpy as np
import torch
from .controller import Controller
import math

"""
cost_func -> states_cost_func
"""
class MPC(Controller):
    def __init__(self, agent, is_torch=True, writer=None):
        
        super(MPC, self).__init__(agent, is_torch, writer)
        env = self.env
        config = self.config
        self.dO, self.dU = env.observation_space.shape[0], env.action_space.high.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low
        self.action_dim = self.ac_ub.shape[0]
        
        actions_num = config.agent.predict_length
        self.actions_num = actions_num

        self.lower_bound = torch.tile(torch.tensor(self.ac_lb), [actions_num]).to(self.config.device)
        self.upper_bound = torch.tile(torch.tensor(self.ac_ub), [actions_num]).to(self.config.device)
        self.prev_sol = (self.lower_bound + self.upper_bound) / 2
        
        Optimizer = optimizers.get_item(config.agent.optimizer)
        self.optimizer = Optimizer(
            sol_dim=actions_num * self.action_dim,
            config=self.config,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound
        )
    
    def reset(self):
        self.prev_sol = (self.lower_bound + self.upper_bound) / 2
        self.net_err = torch.zeros(self.config.agent.ensemble_size, device=self.config.device)
        self.optimizer.reset()
        
    def sample(self, states, epoch=-1, step=-1):
        
        super(MPC, self).sample(states, epoch, step)

        predict_length = self.config.agent.predict_length
        num_particles = self.config.agent.num_particles
        action_dim = self.dU
        def states_cost_func(ac_seqs, return_torch = True, sample_epoch = -1, solution = False):
            if not isinstance(ac_seqs, torch.Tensor):
                ac_seqs = torch.tensor(ac_seqs, device=self.config.device).float()
            batch_size = ac_seqs.shape[0]
            ac_seqs = ac_seqs.reshape(batch_size, predict_length, 1, action_dim)
            ac_seqs = ac_seqs.transpose(0, 1).contiguous()
            ac_seqs = ac_seqs.expand(-1, -1, num_particles, -1)
            return self.mpc_cost_fun(ac_seqs, states, return_torch, sample_epoch=sample_epoch, solution=solution)
        
        opt_action_next = self.optimizer.obtain_solution(
            states_cost_func, 
            self.prev_sol
        )
        
        temp_ac_seqs = opt_action_next.reshape(self.actions_num, action_dim)

        self.prev_sol = torch.cat([
            torch.tensor(temp_ac_seqs[1:], device=self.config.device), 
            torch.zeros([1, action_dim], device=self.config.device)
        ], dim=0).reshape(self.actions_num*action_dim,)

        return temp_ac_seqs[0]