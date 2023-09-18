from .agent import Agent
import mymbrl.controllers as controllers
import mymbrl.models as models
import mymbrl.envs as envs
import mymbrl.dataloaders as dataloaders
import torch, numpy as np
import torch.nn as nn
from scipy.stats import norm
import time
from joblib import Parallel, delayed
import dill
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import parallel_backend
import random
import math
import datetime
import mymbrl.optimizers as optimizers
from mymbrl.utils import shuffle_rows

class DPETS(Agent):

    def __init__(self, config, env, writer):
        """
        Controller: MPC
        """
        self.config = config
        self.env = env
        self.writer = writer
        self.exp_epoch = 0
        
        Model = models.get_item(config.agent.model)
        
        self.model = Model(
            ensemble_size=config.agent.ensemble_size,
            in_features=env.MODEL_IN,
            out_features=env.MODEL_OUT*2,
            hidden_size=config.agent.dynamics_hidden_size, 
            drop_prob=config.agent.dropout,
            device=config.device
        )
        
        self.model = self.model.to(config.device)
        self.dynamics_optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.agent.dynamics_lr, 
            weight_decay=self.config.agent.dynamics_weight_decay
        )
        self.dynamics_scheduler = torch.optim.lr_scheduler.StepLR(
        self.dynamics_optimizer, step_size=1, gamma=config.agent.dynamics_lr_gamma)
        if config.agent.dropout > 0:
            self.model.sample_new_mask(dropout_mask_nums=config.agent.dropout_mask_nums)
            num_particles = self.config.agent.num_particles
            self.model.select_mask(num_particles)

        Dataloader = dataloaders.get_item('free_trend')
        self.dataloader = Dataloader()

        Controller = controllers.get_item('MPC')
        self.controller = Controller(
            self,
            writer=writer
        )

    def reset(self):
        self.controller.reset()
        
    def set_epoch(self, exp_epoch):
        
        self.exp_epoch = exp_epoch
    
    def set_step(self, exp_step):
        self.exp_step = exp_step
        
    def set_model(self, model):
        self.model = model

    def train(self):
        """
        训练一个agent
        """
        dynamics_model = self.model
        for param in dynamics_model.parameters():
            param.requires_grad = True

        num_nets = dynamics_model.num_nets
        num_particles = self.config.agent.num_particles
        net_particles = num_particles // num_nets
        
        dynamics_model.train()
        dynamics_optimizer = self.dynamics_optimizer
        
        self.trainloader = torch.utils.data.DataLoader(
            self.dataloader, 
            batch_size=self.config.agent.train_batch_size, 
            shuffle=True,
            drop_last=True
        )
        
        i = 0
        data_len = self.dataloader.len()
        idxs = np.arange(data_len)
        idxs = idxs.reshape(1,-1)
        idxs = np.tile(idxs, [num_nets, 1])
        
        batch_size = self.config.agent.train_batch_size
        num_batch = idxs.shape[-1] // batch_size

        x_all, y_all, a_all, y2_all, x2_all = self.dataloader.get_x_y_all()

        for _ in range(self.config.agent.train_epoch):
            idxs = shuffle_rows(idxs)
            for batch_num in range(num_batch):
                dynamics_model.select_mask(batch_size)
                batch_idxs = idxs[:, batch_num * batch_size : (batch_num + 1) * batch_size]
                x, y, a, y2, x2 = x_all[batch_idxs, :], y_all[batch_idxs, :], a_all[batch_idxs, :], y2_all[batch_idxs, :], x2_all[batch_idxs, :]
                x, y, a, y2, x2 = x.to(self.config.device), y.to(self.config.device), a.to(self.config.device), y2.to(self.config.device), x2.to(self.config.device)
                
                loss = self.config.agent.dynamics_weight_decay_rate * dynamics_model.compute_decays()
                loss += 0.1*0.005 * (dynamics_model.max_logvar.sum() - dynamics_model.min_logvar.sum())
                mean, logvar = dynamics_model(x, ret_logvar=True)

                inv_var = torch.exp(-logvar)
                mes_loss = ((mean - y) ** 2)
                mes_loss_sum = mes_loss.mean(-1).mean(-1).sum()
                train_losses = mes_loss * inv_var + logvar

                train_losses = train_losses.mean(-1).mean(-1).sum()
                loss += train_losses

                predictions = self.env.obs_postproc(x2, mean)
                predictions_true = self.env.obs_postproc(x2, y)
                predictions_preproc = self.env.obs_preproc(predictions)
                inputs2 = torch.cat((predictions_preproc, a), dim=-1)
                dynamics_model.select_mask(batch_size)
                mean2, logvar2 = dynamics_model(inputs2, ret_logvar=True)
                y_2_true = self.env.obs_postproc(predictions_true, y2)
                new_y2 = self.env.targ_proc(predictions, y_2_true)
                inv_var2 = torch.exp(-logvar2)
                
                mes_loss2 = ((mean2 - new_y2) ** 2)
                mes_loss_sum2 = mes_loss2.mean(-1).mean(-1).sum()
                train_losses2 = mes_loss2 * inv_var2 + logvar2
                train_losses2 = train_losses2.mean(-1).mean(-1).sum()
                loss += train_losses2

                loss.backward()
                nn.utils.clip_grad_norm_(dynamics_model.parameters(), max_norm=20, norm_type=2)
                dynamics_optimizer.step()
                dynamics_optimizer.zero_grad()
                i += 1
                if i % 50 == 0:
                    print(i, 'loss', loss.item(), 'mloss1', mes_loss_sum.item(), 'mloss2', mes_loss_sum2.item())

        if self.exp_epoch in self.config.agent.lr_scheduler:
            self.dynamics_scheduler.step()
        self.model.select_mask(net_particles)
        for param in dynamics_model.parameters():
            param.requires_grad = False
    
    def sample(self, states):

        self.model.eval()
        action = self.controller.sample(states, self.exp_epoch, self.exp_step)
        return action

    def add_data(self, states, actions, indexs=[]):
        assert states.shape[0] == actions.shape[0] + 1
        x = np.concatenate((self.env.obs_preproc(states[:-2]), actions[:-1]), axis=1)
        y = self.env.targ_proc(states[:-2], states[1:-1])
        a = actions[1:]
        y2 = self.env.targ_proc(states[1:-1], states[2:])
        x2 = states[:-2]

        self.dataloader.push(x, y, a, y2, x2)
    
    def prediction(self, states, action, t=0, sample_epoch=0, print_info=False):

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.config.device).float()
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, device=self.config.device).float()
        if(states.dim() == 1):
            states = states.unsqueeze(0).expand(self.config.agent.num_particles, 1).float()
        if(action.dim() == 1):
            action = action.unsqueeze(0).expand(states.shape[0], -1).float()

        proc_obs = self.env.obs_preproc(states)

        proc_obs = self._expand_to_ts_format(proc_obs)
        action = self._expand_to_ts_format(action)

        inputs = torch.cat((proc_obs, action), dim=-1)
        net_particles_batch = inputs.shape[1]
        
        if net_particles_batch != self.model.batch_size:
            self.model.select_mask(net_particles_batch)
        mean, var = self.model(inputs)

        predictions = mean
        predictions = self._flatten_to_matrix(predictions)
        return self.env.obs_postproc(states, predictions)
    
    def _expand_to_ts_format(self, mat):
        dim = mat.shape[-1]
        reshaped = mat.view(-1, self.model.num_nets, self.config.agent.num_particles // self.model.num_nets, dim)
        transposed = reshaped.transpose(0, 1)
        reshaped = transposed.contiguous().view(self.model.num_nets, -1, dim)
        return reshaped
    
    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.shape[-1]
        reshaped = ts_fmt_arr.view(self.model.num_nets, -1, self.config.agent.num_particles // self.model.num_nets, dim)
        transposed = reshaped.transpose(0, 1)
        reshaped = transposed.contiguous().view(-1, dim)
        return reshaped