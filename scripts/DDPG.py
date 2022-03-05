#! /usr/bin/env python
# coding :utf-8
import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.buffer import PrioritizedReplayBuffer

'''
Deep Deterministic Policy Gradients (DDPG)
Original paper:
    CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING https://arxiv.org/abs/1509.02971
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
state_dim = 41
action_dim = 2
tau = 0.01
actor_lr = 3e-4
critic_lr = 3e-4
discount = 0.99
buffer_size = 20000
batch_size = 512
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
        self.l3 = nn.Linear(256, action_dim)

        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))

        return a


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, init_w=3e-3):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        sa = torch.cat((state, action), 1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q


class DDPG:

    def __init__(self, **kwargs):

        # load params
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = PrioritizedReplayBuffer(buffer_size, batch_size, "DDPG")

        self.actor_loss = 0
        self.critic_loss = 0

        self.num_training = 0

        self.load()


    def act(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def put(self, *transition):
        state, action, _, _, _ = transition
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        action = torch.FloatTensor(action).to(device).unsqueeze(0)
        Q = self.critic(state, action).detach()
        self.buffer.add(transition, 1000.0)

        return Q.cpu().item()

    def update(self):

        if not self.buffer.sample_available():
            return

        (state, action, reward, next_state, done), indices = self.buffer.sample()

        # state = (state - self.buffer.state_mean())/(self.buffer.state_std() + 1e-7)
        # next_state = (next_state - self.buffer.state_mean())/(self.buffer.state_std() + 1e-6)
        # reward = reward / (self.buffer.reward_std() + 1e-6)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * discount * target_Q

        # Get current Q estimates
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_loss = critic_loss.item()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # update priorities
        if self.use_priority:
            priorities = (((current_Q - target_Q).detach()**2)).cpu().squeeze(1).numpy() + hyper_parameters_eps        

            self.buffer.update_priorities(indices, priorities)

        # Compute actor losse
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_loss = actor_loss.item()

        if (self.fix_actor_flag == False):
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if (self.fix_actor_flag == False):
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.num_training += 1

    def save(self):
        torch.save(self.critic.state_dict(), url + "DDPG_critic.pth")
        torch.save(self.critic_optimizer.state_dict(), url + "DDPG_critic_optimizer.pth")
        torch.save(self.actor.state_dict(), url + "DDPG_actor.pth")
        torch.save(self.actor_optimizer.state_dict(), url + "DDPG_actor_optimizer.pth")
        self.buffer.save()

    def load(self):
        if self.load_critic_flag == True:
            print("Load critic model.")
            self.critic.load_state_dict(torch.load(url + "DDPG_critic.pth", map_location=device))
            self.critic_target = copy.deepcopy(self.critic)

        if self.load_actor_flag == True:
            print("Load actor model.")
            self.actor.load_state_dict(torch.load(url + "DDPG_actor.pth", map_location=device))
            self.actor_target = copy.deepcopy(self.actor)

        if self.load_optim_flag == True:
            print("Load optimizer.")
            self.critic_optimizer.load_state_dict(torch.load(url + "DDPG_critic_optimizer.pth", map_location=device))
            self.actor_optimizer.load_state_dict(torch.load(url + "DDPG_actor_optimizer.pth", map_location=device))

        if self.load_buffer_flag == True:
            print("Load buffer data.")
            self.buffer.load()