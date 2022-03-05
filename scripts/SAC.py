#! /usr/bin/env python
# coding :utf-8

import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from common.buffer import PrioritizedReplayBuffer

'''
Soft actor critic with automated temperature
Original paper:
    Soft Actor-Critic Algorithms and Applications https://arxiv.org/pdf/1812.05905.pdf
    Learning to Walk via Deep Reinforcement Learning https://arxiv.org/pdf/1812.11103.pdf
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
state_dim = 41
action_dim = 2
tau = 0.01
actor_lr = 3e-4
Q_net_lr = 3e-4
alpha_lr = 3e-4
discount = 0.99
init_temperature = 0.1
buffer_size = 20000
batch_size = 512
actor_update_frequency = 1
hyper_parameters_eps = 0.2
seed = 1

url = os.path.dirname(os.path.realpath(__file__)) + '/data/'

# Set seeds
torch.manual_seed(seed)
np.random.seed(seed)


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, init_w=3e-3):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)

        self.mu_head = nn.Linear(256, action_dim)
        self.mu_head.weight.data.uniform_(-init_w, init_w)
        self.mu_head.bias.data.uniform_(-init_w, init_w)

        self.log_std_head = nn.Linear(256, action_dim)
        self.log_std_head.weight.data.uniform_(-init_w, init_w)
        self.log_std_head.bias.data.uniform_(-init_w, init_w)

        self.min_log_std = -20
        self.max_log_std = 2

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = log_std.exp()
        return mu, std

    def evaluate(self, state):
        mu, std = self.forward(state)
        dist = Normal(0, 1)
        z = dist.sample(mu.shape).to(device)
        # reparameterization trick
        action = torch.tanh(mu + std*z)
        log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True)
        log_prob -= (2*(np.log(2) - action - F.softplus(-2* action))).sum(axis=1, keepdim=True)
        return action, log_prob, z, mu, std

class Q_net(nn.Module):

    def __init__(self, state_dim, action_dim, init_w=3e-3):
        super(Q_net, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.uniform_(-init_w, init_w)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        self.l6.weight.data.uniform_(-init_w, init_w)
        self.l6.bias.data.uniform_(-init_w, init_w)

    def forward(self, s, a):
        sa = torch.cat((s, a), 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, s, a):
        sa = torch.cat((s, a), 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class SAC:

    def __init__(self, **kwargs):

        # load params
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.Q_net = Q_net(state_dim, action_dim).to(device)
        self.Q_net_target = copy.deepcopy(self.Q_net)
        self.Q_net_optimizer = optim.Adam(self.Q_net.parameters(), lr=Q_net_lr)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # set target entropy to -|A|
        self.target_entropy = -action_dim

        self.buffer = PrioritizedReplayBuffer(buffer_size, batch_size, "SAC")
        
        self.actor_loss = 0
        self.critic_loss = 0
        self.alpha_loss = 0

        self.num_training = 0

        self.load()


    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, state):
        state = torch.FloatTensor(np.array(state).reshape(1, -1)).to(device)
        with torch.no_grad():
            mu, std = self.actor(state)
            dist = Normal(mu, std)
            z = dist.sample()
            action = torch.tanh(z)
        return action.flatten().cpu().numpy()

    def put(self, *transition):
        state, action, _, _, _ = transition
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        action = torch.FloatTensor(action).to(device).unsqueeze(0)
        Q = self.Q_net.Q1(state, action).detach()
        self.buffer.add(transition, 1000.0)

        return Q.cpu().item()

    def update(self):

        if not self.buffer.sample_available():
            return

        (state, action, reward, next_state, done), indices = self.buffer.sample()

        # state = (state - self.buffer.state_mean())/(self.buffer.state_std() + 1e-7)
        # next_state = (next_state - self.buffer.state_mean())/(self.buffer.state_std() + 1e-6)
        # reward = reward / (self.buffer.reward_std() + 1e-6)


        # alpha

        current_Q1, current_Q2 = self.Q_net(state, action)
        new_action, log_prob, _, _, _ = self.actor.evaluate(state)
        new_next_action, _, _, _, _ = self.actor.evaluate(next_state)

        if self.num_training % actor_update_frequency == 0 and self.fix_actor_flag == False:

            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_loss = alpha_loss.item()

            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            nn.utils.clip_grad_norm_(self.log_alpha, 0.5)
            self.log_alpha_optimizer.step()

        # Q_net

        target_Q1, target_Q2 = self.Q_net_target(next_state, new_next_action)
        target_Q = reward + (1 - done) * discount * torch.min(target_Q1, target_Q2)
        Q_net_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_loss = Q_net_loss.item()

        self.Q_net_optimizer.zero_grad()
        Q_net_loss.backward()
        nn.utils.clip_grad_norm_(self.Q_net.parameters(), 0.5)
        self.Q_net_optimizer.step()

        # update priorities
        if self.use_priority:
            priorities = (((current_Q1 - target_Q).detach()**2)).cpu().squeeze(1).numpy() \
                        + (((current_Q2 - target_Q).detach()**2)).cpu().squeeze(1).numpy() \
                        + hyper_parameters_eps        

            self.buffer.update_priorities(indices, priorities)        

        # actor
        if self.num_training % actor_update_frequency == 0 and self.fix_actor_flag == False:

                new_Q1, new_Q2 = self.Q_net(state, new_action)
                new_Q = torch.min(new_Q1, new_Q2)
                actor_loss = (self.alpha * log_prob - new_Q).mean()
                self.actor_loss = actor_loss.item()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

        # soft update
        for target_param, param in zip(self.Q_net_target.parameters(), self.Q_net.parameters()):
            target_param.data.copy_(target_param * (1 - tau) + param * tau)

        self.num_training += 1

    def save(self):
        torch.save(self.Q_net.state_dict(), url + "SAC_critic.pth")
        torch.save(self.Q_net_optimizer.state_dict(), url + "SAC_critic_optimizer.pth")
        torch.save(self.actor.state_dict(), url + "SAC_actor.pth")
        torch.save(self.actor_optimizer.state_dict(), url + "SAC_actor_optimizer.pth")
        torch.save(self.log_alpha, url + "SAC_log_alpha.pth")
        torch.save(self.log_alpha_optimizer.state_dict(), url + "SAC_log_alpha_optimizer.pth")
        self.buffer.save()

    def load(self):

        if self.load_critic_flag == True:
            print("Load critic model.")
            self.Q_net.load_state_dict(torch.load(url + "SAC_critic.pth", map_location=device))
            self.Q_net_target = copy.deepcopy(self.Q_net)
        
        if self.load_log_alpha_flag == True:
            print("Load log-alpha.")
            self.log_alpha = torch.load(url + "SAC_log_alpha.pth", map_location=device)
            self.log_alpha.requires_grad = True
            self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        if self.load_actor_flag == True:
            print("Load actor model.")
            self.actor.load_state_dict(torch.load(url + "SAC_actor.pth", map_location=device))
            self.actor_target = copy.deepcopy(self.actor)

        if self.load_optim_flag == True:
            print("Load optimizer.")
            self.Q_net_optimizer.load_state_dict(torch.load(url + "SAC_critic_optimizer.pth", map_location=device))
            self.actor_optimizer.load_state_dict(torch.load(url + "SAC_actor_optimizer.pth", map_location=device))
            self.log_alpha_optimizer.load_state_dict(torch.load(url + "SAC_log_alpha_optimizer.pth", map_location=device))

        if self.load_buffer_flag == True:
            print("Load buffer data.")
            self.buffer.load()