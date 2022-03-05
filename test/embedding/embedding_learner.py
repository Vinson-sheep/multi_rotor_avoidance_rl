#! /usr/bin/env python
# coding :utf-8

import numpy as np
import rospy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from embedding_buffer import EmbeddingBuffer
import os
import scipy.io as sio
import visdom

# parameters
load_able = False

begin_epoch = 1
loss = []
max_epoch = 300

input_dim = 35
output_dim = 35
embedding_lr = 0.01
batch_size = 512
load_embedding_flag =  False
load_embedding_optim_flag = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
url = os.path.dirname(os.path.realpath(__file__))
viz = visdom.Visdom(env="loss")

opts={
    'showlegend': False,
    'title': "epoch-loss",
    'xlabel': "epoch",
    'ylabel': "loss",
}

class Embedding(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Embedding, self).__init__()
        self.l1 = nn.Linear(input_dim, 50)
        self.l2 = nn.Linear(50, 50)
        self.l3 = nn.Linear(50, 30)
        self.l4 = nn.Linear(30, output_dim)
        
    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = torch.relu(self.l3(a))
        a = torch.tanh(self.l4(a))
        
        return a


class Learner(object):

    def __init__(self, **kwargs):

        # load params
        for key, value in kwargs.items():
            setattr(self, key, value)

        # initialize net
        self.embedding = Embedding(self.input_dim, self.output_dim).to(device)
        self.embedding_optimizer = optim.Adam(self.embedding.parameters(), lr = self.embedding_lr, weight_decay=0.01)

        self.buffer = EmbeddingBuffer()

        # load buffer
        self.buffer.load()
        print("load buffer. Size = %d" % (self.buffer.size()))

        # load model and optimizer
        self.load()    


    def learn(self):
        
        # Sample buffer 
        (cur_state, next_state) = self.buffer.sample(self.batch_size)

        cur_state = cur_state[:, 0: 35]
        next_state = next_state[:, 0: 35]

        # Cet target value
        with torch.no_grad():
            target = self.embedding(next_state)            

        # Get current value
        current = self.embedding(cur_state)

        # Compute critic loss
        loss = F.mse_loss(current, target)

        # Optimize the critic
        self.embedding_optimizer.zero_grad()
        loss.backward()
        self.loss = loss.item()
        print("embedding loss: %f" % (loss.item()))
        self.embedding_optimizer.step()

    def last_loss(self):
        return self.loss

    def save(self):

        torch.save(self.embedding.state_dict(), url + "/embedding.pkl")
        torch.save(self.embedding_optimizer.state_dict(), url + "/embedding_optimizer.pth")        

        
    def load(self):

        if self.load_embedding_flag == True:
            print("load embedding model.")
            self.embedding.load_state_dict(torch.load(url + "/embedding.pkl", map_location=device))

        if self.load_embedding_optim_flag == True:
            print("load embedding optimizer.")
            self.embedding_optimizer.load_state_dict(torch.load(url + "/embedding_optimizer.pth", map_location=device))


if __name__ == '__main__':

    import rospy

    # initialize ros
    rospy.init_node("embedding_learner")

    # initialize agent
    kwargs = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'embedding_lr': embedding_lr,
        'batch_size': batch_size,
        'load_embedding_flag': load_embedding_flag,
        'load_embedding_optim_flag': load_embedding_optim_flag,
    } 

    myLearner = Learner(**kwargs)

    if load_able == True:
        loss = list(sio.loadmat(url + 'loss.mat')['data'][0])
        begin_epoch = len(loss) + 1
        print("restore epoch: %d" % (begin_epoch))

    begin_time = rospy.Time.now()

    for epoch in range(begin_epoch, max_epoch + 1):

        print("*"*20 + str(epoch) + "*"*20)

        # learn 
        myLearner.learn()
        loss.append(myLearner.last_loss())

        # save
        myLearner.save()
        sio.savemat(url + '/loss.mat',{'data': loss},True,'5', False, False,'row')

        # plot
        y = loss
        x = range(1, len(loss)+1)
        viz.line(loss, x, win="gazebo1", name="line1", update=None, opts=opts)

    print((rospy.Time.now() - begin_time).to_sec())
                                           
  
