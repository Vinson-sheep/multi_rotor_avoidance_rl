#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

import numpy as np
import os
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrioritizedReplayBuffer:


    def __init__(self, buffer_size, batch_size, policy):
        """
        prioritized replay buffer implemented by Pytorch
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.policy = policy

        self.cur_state_buffer = torch.FloatTensor([]).to(device)
        self.action_buffer = torch.FloatTensor([]).to(device)
        self.reward_buffer = torch.FloatTensor([]).to(device)
        self.next_state_buffer = torch.FloatTensor([]).to(device)
        self.done_buffer =torch.FloatTensor([]).to(device)
        self.priority_buffer = torch.FloatTensor([]).to(device)

        self.url = os.path.dirname(os.path.realpath(__file__)) + '/../data/' + self.policy + '_buffer.pth'


    def add(self, transition, priority):
        """
        arg:
            - transition: the payload needed to be stored
            - prority: the priority of transition
        """

        state, action, reward, next_state, done  = transition

        # pop data
        if self.size() == self.buffer_size:
            self.cur_state_buffer = self.cur_state_buffer[1:]
            self.action_buffer = self.action_buffer[1:]
            self.reward_buffer = self.reward_buffer[1:]
            self.next_state_buffer = self.next_state_buffer[1:]
            self.done_buffer = self.done_buffer[1:]
            self.priority_buffer = self.priority_buffer[1:]

        # push data
        with torch.no_grad():
            self.cur_state_buffer = torch.cat((self.cur_state_buffer, torch.FloatTensor([state]).to(device)), 0)
            self.action_buffer = torch.cat((self.action_buffer, torch.FloatTensor([action]).to(device)), 0)
            self.reward_buffer = torch.cat((self.reward_buffer, torch.FloatTensor([reward]).to(device)), 0)
            self.next_state_buffer = torch.cat((self.next_state_buffer, torch.FloatTensor([next_state]).to(device)), 0)
            self.done_buffer = torch.cat((self.done_buffer, torch.FloatTensor([done]).to(device)), 0)
            self.priority_buffer = torch.cat((self.priority_buffer, torch.FloatTensor([priority]).to(device)), 0)

        return True


    def state_mean_std(self):
        return self.cur_state_buffer.mean(dim=0), self.cur_state_buffer.std(dim=0)

    def sample(self):
        """
        sample from buffer according to priority
        """

        with torch.no_grad():

            size = self.priority_buffer.shape[0]
            possibility = self.priority_buffer / self.priority_buffer.sum()

            indices_cpu = np.random.choice(size, self.batch_size, replace=False, p=possibility.cpu().numpy())
            indices_gpu = torch.LongTensor(indices_cpu).to(device)

            cur_state = self.cur_state_buffer.index_select(0, indices_gpu)
            action = self.action_buffer.index_select(0, indices_gpu)
            reward = self.reward_buffer.index_select(0, indices_gpu).view(self.batch_size, -1)
            next_state = self.next_state_buffer.index_select(0, indices_gpu)
            done = self.done_buffer.index_select(0, indices_gpu).view(self.batch_size, -1)

        return (cur_state, action, reward, next_state, done), indices_cpu


    def update_priorities(self, indices, priority):
        """
        update priorities.
        """
    
        for i, p in zip(indices, torch.FloatTensor(priority)):
            self.priority_buffer[i] = p

        return True


    def sample_available(self):
        """
        True if len(self.memory) >= self.batch_size
        """
        return self.size() >= self.batch_size


    def size(self):
        """
        size of buffer
        """
        return self.priority_buffer.shape[0]


    def save(self):
        """
        save data
        """
        data_dict = {
            "cur_state_buffer": self.cur_state_buffer,
            "action_buffer": self.action_buffer,
            "reward_buffer": self.reward_buffer,
            "next_state_buffer": self.next_state_buffer,
            "done_buffer": self.done_buffer,
            "priority_buffer": self.priority_buffer,
        }

        torch.save(data_dict, self.url)

        print("Saved. Buffer size = %d" % (self.size()))


    def load(self):
        """
        load data
        """

        data_dict = torch.load(self.url)

        self.cur_state_buffer = data_dict["cur_state_buffer"].to(device)
        self.action_buffer = data_dict["action_buffer"].to(device)
        self.reward_buffer = data_dict["reward_buffer"].to(device)
        self.next_state_buffer = data_dict["next_state_buffer"].to(device)
        self.done_buffer = data_dict["done_buffer"].to(device)
        self.priority_buffer = data_dict["priority_buffer"].to(device)

        print("Restore buffer size = %d" % (self.size()))
            

if __name__ == '__main__':

    test_buffer = PrioritizedReplayBuffer(10, 4, "PPO")
    
    print("sample available:", test_buffer.sample_available())

    trans = [
        [[1, 2, 3], 4, 5, [6, 7, 8], 9, 0.1],
        [[10, 11, 12], 13, 14, [15, 16, 17], 18, 0.2],
        [[19, 20, 21], 22, 23, [24, 25, 26], 27, 0.3],
        [[28, 29, 30], 31, 32, [33, 34, 35], 36, 0.4],
        [[37, 38, 39], 40, 41, [42, 43, 44], 45, 0.5],
    ]

    priority = [
        101,
        102,
        103,
        104,
        105
    ]

    for i in range(20):
        for j in range(5):
            test_buffer.add(trans[j], priority[j])

    # print(test_buffer.cur_state_buffer)
    # print(test_buffer.action_buffer)
    # print(test_buffer.reward_buffer)
    # print(test_buffer.next_state_buffer)
    # print(test_buffer.done_buffer)
    print(test_buffer.priority_buffer)

    samples, indices = test_buffer.sample()
    print(samples)
    print(indices)

    test_buffer.update_priorities(indices, [1, 2, 3, 5])

    # print(test_buffer.cur_state_buffer)
    # print(test_buffer.action_buffer)
    # print(test_buffer.reward_buffer)
    # print(test_buffer.next_state_buffer)
    # print(test_buffer.done_buffer)
    print(test_buffer.priority_buffer)

    print("sample available:", test_buffer.sample_available())

    test_buffer.save()

    test_buffer.load()





    

    
