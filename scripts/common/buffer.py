#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

from cgi import test

from matplotlib.style import available
import numpy as np
import os
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

available_num = 5

class ReplayBufferLSTM:

    def __init__(self, capacity, policy):
        """
        Replay buffer for agent with LSTM network additionally using previous action, can be used 
        if the hidden states are not stored (arbitrary initialization of lstm for training).
        And each sample contains the whole episode instead of a single step.
        """
        self.capacity = capacity
        self.policy = policy
        self.position = 0

        self.buffer = []

        self.url = os.path.dirname(os.path.realpath(__file__)) + '/../data/' + self.policy + '_buffer.pth'


    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done):
        """
        arg:
            - transition: the payload needed to be stored
            - prority: the priority of transition
        """

        # batch first configuration
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        action = torch.FloatTensor(action).to(device).unsqueeze(0)
        last_action = torch.FloatTensor(last_action).to(device).unsqueeze(0)
        reward = torch.FloatTensor(reward).to(device).view(-1, 1).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).to(device).unsqueeze(0)
        done = torch.FloatTensor(done).to(device).view(-1, 1).unsqueeze(0)

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (hidden_in, hidden_out, state, action, last_action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sampleSingle(self):
        
        index = random.randint(0, len(self.buffer)-1)

        return self.buffer[index]

    def sample_available(self):
        return self.size() >= available_num

    def size(self):
        return len(self.buffer)

    def bufferLen(self):
        pass

    def save(self):
        """
        save data
        """
        data_dict = {
            "buffer": self.buffer,
            "position": self.position
        }

        torch.save(data_dict, self.url)

        print("Saved. Buffer size = %d" % (self.size()))


    def load(self):
        """
        load data
        """

        data_dict = torch.load(self.url)
        
        self.buffer = data_dict["buffer"]
        self.position = data_dict["position"]

        print("Restore buffer size = %d" % (self.size()))


if __name__ == '__main__':

    test_buffer = ReplayBufferLSTM(10, "SAC")

    hidden_in = (torch.rand(4), torch.rand(4))
    hidden_out = (torch.rand(4), torch.rand(4))

    state = []
    next_state = []
    action = []
    last_action = []
    reward = []
    done = []

    for i in range(0, 2):
        state.append(torch.rand(3).tolist()) 
        next_state.append(torch.rand(3).tolist())
        action.append(torch.rand(2).tolist())
        last_action.append(torch.rand(2).tolist())
        reward.append(torch.rand(1)[0].item())
        done.append(torch.rand(1)[0].item())
    
    test_buffer.push(hidden_in, hidden_out, state, action, last_action, reward, next_state, done)

    test_buffer.save()
    test_buffer.load()
    


    




    

    
