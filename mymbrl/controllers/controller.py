import torch
import numpy as np

class Controller:
    def __init__(self, agent, is_torch=True, writer=None):
        self.config = agent.config
        self.is_torch = is_torch
        self.agent = agent
        self.prediction = agent.prediction
        self.env = agent.env
        self.dataloader = agent.dataloader
        self.writer = writer
        
    def set_epoch(self, exp_epoch):
        self.exp_epoch = exp_epoch
        
    def set_step(self, exp_step):
        self.exp_step = exp_step
        
    def sample(self, states, epoch=-1, step=-1):
        self.epoch = epoch
        self.step = step
    
    def mpc_cost_fun(self, ac_seqs, cur_obs, return_torch = True, sample_epoch = -1, solution=False):
        # (400, 25, 20, 4)
        plt_info = self.config.experiment.plt_info
        nopt = ac_seqs.shape[1]
        npart = self.config.agent.num_particles

        # if not isinstance(ac_seqs, torch.Tensor):
        #     ac_seqs = torch.from_numpy(ac_seqs).float().to(self.config.device)
        
        if not isinstance(cur_obs, torch.Tensor):
            cur_obs = torch.from_numpy(cur_obs).float().to(self.config.device)
        obs_dim = cur_obs.ndim
        if obs_dim == 2 and cur_obs.shape[0] == npart:
            cur_obs = cur_obs[None]
            cur_obs = cur_obs.expand(nopt, -1, -1).reshape(nopt * npart, -1)
        elif obs_dim == 1:
            cur_obs = cur_obs[None]
            cur_obs = cur_obs.expand(nopt * npart, -1)
        else:
            raise ValueError("step states shape error!")

        costs = torch.zeros(nopt, npart, device=self.config.device)
        if solution and plt_info:
            stds = np.zeros(cur_obs.shape[-1])
        for t in range(self.config.agent.predict_length):
            cur_acs = ac_seqs[t]
            next_obs = self.prediction(cur_obs, cur_acs, t, sample_epoch, print_info=solution)
            cost = self.env.obs_cost_fn_cost(next_obs) + self.env.ac_cost_fn_cost(cur_acs.reshape(-1, cur_acs.shape[-1]))
            cost = cost.view(-1, npart)
            costs += cost
            cur_obs = next_obs
            # 记录预测值
            if solution and t == 0:
                self.pre_pred_obs = next_obs.detach().clone()
            if solution and plt_info:
                std = torch.std(cur_obs, dim=0).cpu().numpy()
                mean = torch.mean(cur_obs, dim=0).cpu().numpy()
                stds += std
                if t == 0:
                    # std_list = std.tolist()
                    for dim in range(std.shape[0]):
                        self.writer.add_scalar('plt_mbrl/one_step_std/epoch'+str(self.epoch)+'/dim'+str(dim), std[dim], self.step)
                        self.writer.add_scalar('plt_mbrl/one_step_mean/epoch'+str(self.epoch)+'/dim'+str(dim), mean[dim], self.step)
        if solution and plt_info:
            std = stds / self.config.agent.predict_length
            # std_list = std.tolist()
            for dim in range(std.shape[0]):
                self.writer.add_scalar('plt_mbrl/all_step_std/epoch'+str(self.epoch)+'/dim'+str(dim), std[dim], self.step)
        # self.writer.add_scalar('plt_mbrl/rewards/epoch'+str(self.epoch)+'/all_step_std', position_x, self.step)
        costs[costs != costs] = 1e6

        if return_torch:
            return costs.mean(dim=1)
        return costs.mean(dim=1).detach().cpu().numpy()
    
    def mpc_done_cost_fun(self, ac_seqs, cur_obs, return_torch = True, sample_epoch = -1, solution=False):
        nopt = ac_seqs.shape[1]
        npart = self.config.agent.num_particles

        if not isinstance(cur_obs, torch.Tensor):
            cur_obs = torch.from_numpy(cur_obs).float().to(self.config.device)
        obs_dim = cur_obs.ndim
        if obs_dim == 2 and cur_obs.shape[0] == npart:
            cur_obs = cur_obs[None]
            cur_obs = cur_obs.expand(nopt, -1, -1).reshape(nopt * npart, -1)
        elif obs_dim == 1:
            cur_obs = cur_obs[None]
            cur_obs = cur_obs.expand(nopt * npart, -1)
        else:
            raise ValueError("step states shape error!")

        costs = torch.zeros(nopt, npart, device=self.config.device)
        pre_obs_info = None
        for t in range(self.config.agent.predict_length):
            cur_acs = ac_seqs[t]
            next_obs = self.prediction(cur_obs, cur_acs, t, sample_epoch, print_info=solution)
            obs_cost, pre_obs_info = self.env.obs_cost_fn_cost_done(next_obs, t, pre_obs_info)
            cost = obs_cost + self.env.ac_cost_fn_cost(cur_acs.reshape(-1, cur_acs.shape[-1]))
            cost = cost.view(-1, npart)
            costs += cost
            cur_obs = next_obs
            if solution and t == 0:
                self.pre_pred_obs = next_obs.detach().clone()
        costs[costs != costs] = 1e6

        if return_torch:
            return costs.mean(dim=1)
        return costs.mean(dim=1).detach().cpu().numpy()
    