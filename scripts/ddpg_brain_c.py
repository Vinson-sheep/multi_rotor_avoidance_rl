#! /usr/bin/env python
# coding :utf-8

__author__ = 'zhenhang.sun@gmail.com'
__version__ = '1.0.0'

# import gym
import math
import random
import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.prioritized_replay_buffer import PrioritizedReplayBuffer
import os

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(35+4, 500)
        self.linear2 = nn.Linear(500, 500)
        self.linear3 = nn.Linear(500, 500)
        self.linear_vx = nn.Linear(500, 1)
        self.linear_yaw = nn.Linear(500, 1)
        
    def forward(self, s):
        x = torch.relu(self.linear1(s))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        vx = torch.sigmoid(self.linear_vx(x))
        yaw = torch.tanh(self.linear_yaw(x))

        x = torch.cat([vx, yaw], 1)
        
        return x


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(35+4, 500)
        self.linear2 = nn.Linear(500+2, 500)
        self.linear3 = nn.Linear(500, 500)
        self.linear4 = nn.Linear(500, 1)

    def forward(self, s, a):
        x = F.relu(self.linear1(s))
        x = torch.cat([x, a], 1)
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)

        return x


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.actor = Actor().cuda()
        self.actor_target = Actor().cuda()
        self.critic = Critic().cuda()
        self.critic_target = Critic().cuda()
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.buffer = PrioritizedReplayBuffer(self.buffer_size, self.batch_size)

        # load and save
        self.actor_load_url = os.path.dirname(os.path.realpath(__file__)) + "/ddpg_data/actor_model.pkl"
        self.critic_load_url = os.path.dirname(os.path.realpath(__file__)) + "/ddpg_data/critic_model.pkl"
        self.actor_save_url = os.path.dirname(os.path.realpath(__file__)) + "/ddpg_data/actor_model.pkl"
        self.critic_save_url = os.path.dirname(os.path.realpath(__file__)) + "/ddpg_data/critic_model.pkl"

        if self.load_data == True:
            self.load_model()

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        
    def act(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float).cuda().unsqueeze(0)
        a0 = self.actor(s0).detach().cpu().squeeze(0).numpy()
        return a0
    
    def put(self, *transition): 
        """
        return Q_value of s0, a0
        """

        s0, a0, r1, s1, done = transition

        s0 = torch.tensor(s0, dtype=torch.float).cuda().unsqueeze(0)
        a0 = torch.tensor(a0, dtype=torch.float).cuda().unsqueeze(0)
        r1 = torch.tensor(r1, dtype=torch.float).view(1,-1).cuda()
        s1 = torch.tensor(s1, dtype=torch.float).cuda().unsqueeze(0)
        done = torch.tensor(done, dtype=torch.float).cuda().view(1,-1).cuda()

        a1 = self.actor_target(s1).detach()
        y_true = r1 + self.gamma * self.critic_target(s1, a1).mul(1-done).detach()
        y_pred = self.critic_target(s0, a0).detach()

        # loss_fn = nn.MSELoss()
        # loss = loss_fn(y_pred, y_true).detach()

        # priority = (loss.item())**self.alpha

        self.buffer.add(transition, 10000.0)

        return y_pred.cpu().item()

    def learn(self):
        if not self.buffer.sample_available():
            return

        samples, indices = self.buffer.sample()

        s0, a0, r1, s1, done = zip(*samples)

        s0 = torch.tensor(s0, dtype=torch.float).cuda()
        a0 = torch.tensor(a0, dtype=torch.float).cuda()
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size,-1).cuda()
        s1 = torch.tensor(s1, dtype=torch.float).cuda()
        done = torch.tensor(done, dtype=torch.float).cuda().view(self.batch_size,-1).cuda()

        def critic_learn():
            a1 = self.actor_target(s1).detach()
            y_true = r1 + self.gamma * self.critic_target(s1, a1).mul(1-done).detach()
            
            y_pred = self.critic(s0, a0)
            
            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

            # update priorities
            priorities = (((y_true - y_pred).detach()**2)*self.alpha).cpu().squeeze(1).numpy() + self.hyper_parameters_eps
            self.buffer.update_priorities(indices, priorities)
        

        def actor_learn():
            loss = -torch.mean( self.critic(s0, self.actor(s0)) )
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
                                           
        def soft_update(net_target, net, tau):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)
    
    def save_data(self):
        torch.save(self.actor, self.actor_save_url)
        torch.save(self.critic, self.critic_save_url)
        self.buffer.save()

    def load_model(self):
        self.actor = torch.load(self.actor_load_url)
        self.critic = torch.load(self.critic_load_url)
        self.buffer.load( )

        
                                           
  
