import mymbrl.agents as agents
import mymbrl.envs as envs
import numpy as np
import gym
import torch
import os

class MBRL:
    def __init__(self, config, writer):
        Env = envs.get_item(config.env)
        env = Env()
        self.env = env
        self.writer = writer
        
        Agent = agents.get_item(config.agent.name)
        self.agent = Agent(config, env, writer)
        self.config = config
        
        self.env.seed(config.random_seed)

    def run(self):
        random_ntrain_iters = self.config.experiment.random_ntrain_iters
        for i in range(random_ntrain_iters):
            rewards = self.one_exp(epoch=i-random_ntrain_iters+1, is_random = True)
            self.writer.add_scalar('mbrl/rewards', rewards.sum(), 0)
        
        for i in range(self.config.experiment.ntrain_iters):
            rewards = self.one_exp(i+1)
            print("epoch", i, "rewards", rewards.sum())
            self.writer.add_scalar('mbrl/rewards', rewards.sum(), i+1)
    
    def one_exp(self, epoch = 0, is_random = False):

        self.agent.set_epoch(epoch)
        self.agent.reset()
        if epoch > 0:
            self.agent.train()

        cur_states = self.env.reset()
        actions = []
        rewards = []
        states = [cur_states]
        for step in range(self.config.experiment.horizon):
            self.agent.set_step(step)
            action = None
            if is_random:
                action = self.env.action_space.sample()
            else:
                action = self.agent.sample(cur_states)
            next_state, reward, done, info= self.env.step(action)
            
            self.writer.add_scalar('mbrl/rewards/epoch'+str(epoch), reward, step)
            
            actions.append(action)
            states.append(next_state)
            rewards.append(reward)
            
            cur_states = next_state

        states, actions, rewards = tuple(map(lambda l: np.stack(l, axis=0),
                                            (states, actions, rewards)))
        
        self.agent.add_data(states, actions)
        return rewards
        
