#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

import collections
import numpy as np
import os
import numpy as np
import pickle

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
        print("buffer size:", len(self.memory))

        save_file = open(os.path.dirname(os.path.realpath(__file__)) + '/temp.bin',"wb")
        pickle.dump(self.memory,save_file)
        pickle.dump(self.priority_tree,save_file)
        save_file.close()
        

    def load(self):

        load_file = open(os.path.dirname(os.path.realpath(__file__)) + '/temp.bin',"rb")
        self.memory=pickle.load(load_file)
        self.priority_tree=pickle.load(load_file)

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





    

    
