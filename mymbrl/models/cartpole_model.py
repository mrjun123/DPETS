import numpy as np
import gym
import torch
from torch import nn as nn
from torch.nn import functional as F
from mymbrl.utils import swish, get_affine_params
import random
import math

class CartpoleModel(nn.Module):

    def __init__(self, ensemble_size, in_features, out_features, hidden_size=200,drop_prob=0.1, dropout_mask_nums=20, device="cpu"):
        super().__init__()

        self.fit_input = False
        self.dropout = False
        self.mask_batch_size = 1
        self.batch_size = 30

        self.num_nets = ensemble_size

        self.drop_prob = drop_prob
        self.dropout_mask_nums = dropout_mask_nums
        self.hidden_size = hidden_size

        self.hidden1_mask = None
        self.hidden2_mask = None
        self.hidden3_mask = None
        self.hidden4_mask = None
        self.hidden5_mask = None

        self.hidden_mask_indexs = None
        self.hidden1_mask_select = None
        self.hidden2_mask_select = None
        self.hidden3_mask_select = None
        self.hidden4_mask_select = None
        self.hidden5_mask_select = None

        self.in_features = in_features
        self.out_features = out_features

        self.lin0_w, self.lin0_b = get_affine_params(ensemble_size, in_features, hidden_size)

        self.lin1_w, self.lin1_b = get_affine_params(ensemble_size, hidden_size, hidden_size)

        self.lin2_w, self.lin2_b = get_affine_params(ensemble_size, hidden_size, hidden_size)
        
        self.lin5_w, self.lin5_b = get_affine_params(ensemble_size, hidden_size, out_features)

        self.inputs_mu = nn.Parameter(torch.zeros(in_features).to(device), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(in_features).to(device), requires_grad=False)

        self.max_logvar = nn.Parameter(torch.ones(1, out_features // 2, dtype=torch.float32).to(device) / 2.0)
        self.min_logvar = nn.Parameter(- torch.ones(1, out_features // 2, dtype=torch.float32).to(device) * 10.0)
    
    def compute_decays(self):

        lin0_decays = 0.0001 * (self.lin0_w ** 2).sum() / 2.0
        lin1_decays = 0.00025 * (self.lin1_w ** 2).sum() / 2.0
        lin2_decays = 0.00025 * (self.lin2_w ** 2).sum() / 2.0
        lin5_decays = 0.0005 * (self.lin5_w ** 2).sum() / 2.0

        return lin0_decays + lin1_decays + lin2_decays + lin5_decays

    def forward(self, inputs, ret_logvar=False, open_dropout=True):

        if self.fit_input:
            inputs = (inputs - self.inputs_mu) / self.inputs_sigma
        inputs = inputs.matmul(self.lin0_w) + self.lin0_b
        inputs = swish(inputs)
        if self.dropout and open_dropout:
            inputs = inputs * self.hidden1_mask_select
        inputs = inputs.matmul(self.lin1_w) + self.lin1_b
        inputs = swish(inputs)
        if self.dropout and open_dropout:
            inputs = inputs * self.hidden2_mask_select
        inputs = inputs.matmul(self.lin2_w) + self.lin2_b
        inputs = swish(inputs)
        if self.dropout and open_dropout:
            inputs = inputs * self.hidden3_mask_select

        inputs = inputs.matmul(self.lin5_w) + self.lin5_b

        mean = inputs[:, :, :self.out_features // 2]

        logvar = inputs[:, :, self.out_features // 2:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar

        return mean, torch.exp(logvar)
    
    def select_mask(self, batch_size=None):
        if not self.dropout:
            return
        self.batch_size = batch_size
        index_list = list(range(0, self.dropout_mask_nums))
        index_tile_list = index_list*math.ceil(batch_size / self.dropout_mask_nums)
        device = self.get_param_device()
        indexs = torch.tensor(random.sample(index_tile_list, batch_size), device=device)
        self.hidden1_mask_select = torch.index_select(self.hidden1_mask, 1, indexs)
        self.hidden2_mask_select = torch.index_select(self.hidden2_mask, 1, indexs)
        self.hidden3_mask_select = torch.index_select(self.hidden3_mask, 1, indexs)

    def sample_new_mask(self, dropout_mask_nums=None, num_particles=None):
        """Sample a new mask for MC-Dropout. Rather than sample the mask at each forward pass
        (as traditionally done in dropout), keep the dropped nodes fixed until this function is
        explicitely called.
        """
        self.dropout = True
        drop_prob = self.drop_prob
        device = self.get_param_device()
        if dropout_mask_nums:
            self.dropout_mask_nums = dropout_mask_nums
        self.hidden1_mask = torch.bernoulli(
            torch.ones(self.num_nets, dropout_mask_nums, self.hidden_size) * (1 - drop_prob)).to(device)
        self.hidden2_mask = torch.bernoulli(
            torch.ones(self.num_nets, dropout_mask_nums, self.hidden_size) * (1 - drop_prob)).to(device)
        self.hidden3_mask = torch.bernoulli(
            torch.ones(self.num_nets, dropout_mask_nums, self.hidden_size) * (1 - drop_prob)).to(device)
    
    def get_param_device(self):
        return next(self.parameters()).device