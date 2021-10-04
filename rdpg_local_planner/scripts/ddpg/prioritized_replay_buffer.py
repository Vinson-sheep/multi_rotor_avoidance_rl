#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

import random
from collections import deque
import numpy as np

class PrioritizedReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        """
        
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.memory = deque(maxlen=self.buffer_size)
        self.priority_tree = deque(maxlen=self.buffer_size)

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





    

    
