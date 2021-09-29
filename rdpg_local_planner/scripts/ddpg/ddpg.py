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

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(77, 500)
        self.linear2 = nn.Linear(500, 500)
        self.linear3 = nn.Linear(500, 500)
        self.linear_vx = nn.Linear(500, 1)
        self.linear_vy = nn.Linear(500, 1)
        self.linear_yaw = nn.Linear(500, 1)

        
    def forward(self, s):
        x = torch.relu(self.linear1(s))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        vx = torch.sigmoid(self.linear_vx(x))
        vy = torch.tanh(self.linear_vy(x))
        yaw = torch.tanh(self.linear_yaw(x))

        x = torch.cat([vx, vy], 1)
        x = torch.cat([x, yaw], 1)
        

        return x


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(77, 500)
        self.linear2 = nn.Linear(500+3, 500)
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

        s_dim = 77
        a_dim = 3

        self.actor = Actor().cuda()
        self.actor_target = Actor().cuda()
        self.critic = Critic().cuda()
        self.critic_target = Critic().cuda()
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.buffer = []
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
    def act(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float).cuda().unsqueeze(0)
        a0 = self.actor(s0).detach().cpu().squeeze(0).numpy()
        return a0
    
    def put(self, *transition): 
        if len(self.buffer)== self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return 
        
        samples = random.sample(self.buffer, self.batch_size)

        s0, a0, r1, s1 = zip(*samples)

        s0 = torch.tensor(s0, dtype=torch.float).cuda()
        a0 = torch.tensor(a0, dtype=torch.float).cuda()
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size,-1).cuda()
        s1 = torch.tensor(s1, dtype=torch.float).cuda()

        def critic_learn():
            a1 = self.actor_target(s1).detach()
            y_true = r1 + self.gamma * self.critic_target(s1, a1).detach()
            
            y_pred = self.critic(s0, a0)
            
            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()
            
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
                                           
                                           
  
