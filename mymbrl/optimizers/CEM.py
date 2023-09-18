from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import scipy.stats as stats
import torch


class CEMOptimizer():

    def __init__(self, sol_dim, config,
                 upper_bound=None, lower_bound=None):

        self.sol_dim = sol_dim
        self.config = config

        self.max_iters = config.agent.CEM.max_iters
        self.popsize = config.agent.CEM.popsize
        self.num_elites = config.agent.CEM.num_elites
        self.alpha = config.agent.CEM.alpha
        self.epsilon = config.agent.CEM.epsilon
        self.max_value = config.agent.CEM.max_value
        self.min_value = config.agent.CEM.min_value

        self.ub, self.lb = upper_bound.cpu().numpy(), lower_bound.cpu().numpy()

        self.init_var = np.square(self.ub - self.lb) / 16

        if self.num_elites > self.popsize:
            raise ValueError("Number of elites must be at most the population size.")

    def reset(self):
        pass

    def obtain_solution(self, cost_function, init_mean):
        init_mean = init_mean.cpu().numpy()
        mean, var, t = init_mean, self.init_var, 0
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))

        all_min = 1e4
        all_min_sol = init_mean
        
        while (t < self.max_iters) and np.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)
            samples = X.rvs(size=[self.popsize, self.sol_dim]) * np.sqrt(constrained_var) + mean
            samples = samples.astype(np.float32)
            
            with torch.no_grad():
                costs = cost_function(samples, return_torch = False)

            costs[costs < self.min_value] = self.max_value
            costs_argsort = np.argsort(costs)
            elites = samples[costs_argsort][:self.num_elites]
            if self.config.agent.CEM.select_min:
                min_cost = costs[costs_argsort[0]]
                if min_cost < all_min:
                    all_min = min_cost
                    all_min_sol = samples[costs_argsort[0]]

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            t += 1
            if self.config.agent.CEM.print:
                print_cost = cost_function(mean[None], return_torch = False).mean().item()
                print("CEM iters", t, "cost", print_cost)
        if self.config.agent.CEM.select_min:
            return all_min_sol
        return mean

