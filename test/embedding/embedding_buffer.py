#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

import numpy as np
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbeddingBuffer:


    def __init__(self):
        """
        Embedding buffer is a container storing state data. The implementation here is 
        for embedding layer constructing.
        """
        self.cur_state_buffer = torch.FloatTensor([]).to(device)
        self.next_state_buffer = torch.FloatTensor([]).to(device)

        self.url = os.path.dirname(os.path.realpath(__file__)) + '/EBuffer.pth'


    def add(self, cur_state, next_state):
        """
        add data to buffer
        """

        # push data
        with torch.no_grad():
            self.cur_state_buffer = torch.cat((self.cur_state_buffer, torch.FloatTensor([cur_state]).to(device)), 0)
            self.next_state_buffer = torch.cat((self.next_state_buffer, torch.FloatTensor([next_state]).to(device)), 0)

        return True


    def sample(self, batch_size):
        """
        sample from buffer according to priority
        """

        with torch.no_grad():

            size = self.cur_state_buffer.shape[0]

            indices_cpu = np.random.choice(size, batch_size, replace=False)
            indices_gpu = torch.LongTensor(indices_cpu).to(device)

            cur_state = self.cur_state_buffer.index_select(0, indices_gpu)
            next_state = self.next_state_buffer.index_select(0, indices_gpu)

        return (cur_state, next_state)


    def size(self):
        """
        size of buffer
        """
        return self.cur_state_buffer.shape[0]


    def save(self):
        """
        save data
        """
        data_dict = {
            "cur_state_buffer": self.cur_state_buffer,
            "next_state_buffer": self.next_state_buffer,
        }

        torch.save(data_dict, self.url)

        print("save buffer size: %d" % (self.size()))


    def load(self):
        """
        load data
        """

        data_dict = torch.load(self.url)

        self.cur_state_buffer = data_dict["cur_state_buffer"].to(device)
        self.next_state_buffer = data_dict["next_state_buffer"].to(device)

        print("restore buffer size: %d" % (self.size()))
            

if __name__ == '__main__':

    test_buffer = EmbeddingBuffer()
    
    print("sample available:", test_buffer.size())

    for i in range(100):
        test_buffer.add(np.random.randn(4), np.random.randn(4))

    print(test_buffer.size())

    (cur_state, next_state) = test_buffer.sample(4)

    print(cur_state)
    print(next_state)


    print("sample available:", test_buffer.size())

    test_buffer.save()

    test_buffer.load()





    

    
