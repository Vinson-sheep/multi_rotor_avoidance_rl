#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

import random
import collections
import numpy as np
import os
import json
import numpy as np

class PrioritizedReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        """
        
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.memory = collections.deque(maxlen=self.buffer_size)
        self.priority_tree = collections.deque(maxlen=self.buffer_size)

    def add(self, transition, priority):
        """
        arg:
            - transition: the payload needed to be stored
            - prority: the priority of transition
        """
        self.memory.append(transition)
        self.priority_tree.append(priority)

        return True

    def sample(self):
        """
        sample from buffer according to priority
        """
        assert len(self.memory) >= self.batch_size, "length of memory is lower than batch size."

        sum_priority = sum(self.priority_tree)
        possibility = np.array(self.priority_tree)/sum_priority
        indices = np.random.choice(len(self.priority_tree), self.batch_size, replace=False, p=possibility)

        samples = np.array(self.memory)[indices]

        return samples, indices

    def update_priorities(self, indices, priority):
        """
        update priorities.
        """
        assert len(indices) == len(priority), "size of indices should match that of priority."
        for i, p in zip(indices, priority):
            self.priority_tree[i] = p

        return True

    def sample_available(self):
        """
        True if len(self.memory) >= self.batch_size
        """
        return len(self.memory) >= self.batch_size

    def save(self):

        # encode
        m = []
        for i in range(len(self.memory)-1, -1, -1):
            t = list(self.memory[i])
            t[0] = np.array(t[0], dtype='float32').tolist()
            t[1] = np.array(t[1], dtype='float32').tolist()
            t[2] = float(t[2])
            t[3] = np.array(t[3], dtype='float32').tolist()
            t[4] = float(t[4])
            m.append(t)

        p  = list(self.priority_tree)

        print("buffer size:", len(self.memory))

        with open(os.path.dirname(os.path.realpath(__file__)) + '/memory.json','w') as file_obj:
            json.dump(m, file_obj)

        with open(os.path.dirname(os.path.realpath(__file__)) + '/priority.json','w') as file_obj:
            json.dump(p, file_obj)



    def load(self):

        with open(os.path.dirname(os.path.realpath(__file__)) + '/memory.json') as file_obj:
            m = json.load(file_obj)

        with open(os.path.dirname(os.path.realpath(__file__)) + '/priority.json') as file_obj:
            p = json.load(file_obj)

        for i,j in zip(m, p):
            self.add(i, j)

        print("restore buffer size:", len(self.memory))
            

if __name__ == '__main__':
    test_buffer = PrioritizedReplayBuffer()
    
    print("sample available:", test_buffer.sample_available())

    for i in range(200):
        test_buffer.add(i, float(i)/200.0)

    # print(test_buffer.memory)
    # print(test_buffer.priority_tree)

    samples, indices = test_buffer.sample()
    print(samples)
    print(indices)

    test_buffer.update_priorities(indices, [1, 2, 3])

    print(test_buffer.memory)
    print(test_buffer.priority_tree)

    print("sample available:", test_buffer.sample_available())





    

    
