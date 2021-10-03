#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

import game
import rospy
import numpy as np
from ddpg_PER import Agent
import visdom
import scipy.io as sio
import os

epsilon = 0.9
epsilon_decay = 0.999

x = []
y = []
r = []
s = []
q = []
step_count = 0
agent = None

viz = visdom.Visdom(env="line")

def limit(x, max_x, min_x):
    return max(min(x, max_x), min_x)

def plotCB(event):
    # save value
    sio.savemat(os.path.dirname(os.path.realpath(__file__)) + '/step.mat',{'data': s},True,'5', False, False,'row')
    sio.savemat(os.path.dirname(os.path.realpath(__file__)) + '/reward.mat',{'data': r},True,'5', False, False,'row')
    sio.savemat(os.path.dirname(os.path.realpath(__file__)) + '/q_value.mat',{'data': q},True,'5', False, False,'row')
    sio.savemat(os.path.dirname(os.path.realpath(__file__)) + '/episode.mat',{'data': x},True,'5', False, False,'row')
    sio.savemat(os.path.dirname(os.path.realpath(__file__)) + '/total_reward.mat',{'data': y},True,'5', False, False,'row')

    # plot
    viz.line(
        y,
        x,
        win="gazebo1",
        name="line1",
        update=None,
        opts={
            'showlegend': True,
            'title': "reward-episode",
            'xlabel': "episode",
            'ylabel': "reward",
        },
    )
    viz.line(
        r,
        s,
        win="gazebo2",
        name="line2",
        update=None,
        opts={
            'showlegend': True,
            'title': "reward-step",
            'xlabel': "step",
            'ylabel': "reward",
        },
    )
    viz.line(
        q,
        s,
        win="gazebo3",
        name="line3",
        update=None,
        opts={
            'showlegend': True,
            'title': "q_value-step",
            'xlabel': "step",
            'ylabel': "Q value",
        },
    )

    # save model
    agent.save_data()

if __name__ == '__main__':
    rospy.init_node("test")
    # global env
    env = game.Game("iris_0")
    
    # load data
    # annotate if it is no need to load data
    # x = sio.loadmat(os.path.dirname(os.path.realpath(__file__)) + '/episode.mat')['data'].toarray()
    # y = sio.loadmat(os.path.dirname(os.path.realpath(__file__)) + '/total_reward.mat')['data'].toarray()
    # r = list(sio.loadmat(os.path.dirname(os.path.realpath(__file__)) + '/reward.mat')['data'])
    # s = list(sio.loadmat(os.path.dirname(os.path.realpath(__file__)) + '/step.mat')['data'])
    # q = list(sio.loadmat(os.path.dirname(os.path.realpath(__file__)) + '/q_value.mat')['data'])
    # if len(r) > 0:
    #     step_count = r[-1]


    # plotTimer = rospy.Timer(rospy.Duration(60), plotCB)
    

    params = {
        'env': env,
        'gamma': 0.99,
        'actor_lr': 0.001,
        'critic_lr': 0.001,
        'tau': 0.02,
        'capacity': 100000,
        'batch_size': 256,
    }

    agent = Agent(**params)

    for episode in range(1000):
        if episode == 0:
            s0 = env.start()
            print("start!")
        else:
            s0 = env.reset()
            print("reset.")

        episode_reward = 0

        for step in range(300):
            step_count += 1
            s.append(step_count)

            a0 = agent.act(s0)

            # if epsilon > np.random.random():
            #     a0[0] += np.random.random()*0.4
            #     a0[1] += (np.random.random()-0.5)*0.5
            #     a0[2] += (np.random.random()-0.5)*0.4

            #     a0[0] = limit(a0[0], 1.0, 0.0)
            #     a0[1] = limit(a0[1], 1.0, -1.0)
            #     a0[2] = limit(a0[2], 1.0, -1.0)


            if epsilon > np.random.random():
                a0[0] += np.random.random()*0.4
                a0[1] = 0
                a0[2] = s0[-1]

                a0[0] = limit(a0[0], 1.0, 0.0)
                a0[1] = limit(a0[1], 1.0, -1.0)
                a0[2] = limit(a0[2], 1.0, -1.0)

            epsilon = max(epsilon_decay*epsilon, 0.10)
            

            print("eps = ", epsilon)

            begin_time = rospy.Time.now()
            s1, r1, done = env.step(0.1, a0[0], a0[1], a0[2])
            q_value = agent.put(s0, a0, r1, s1, done)
            
            r.append(r1)
            q.append(q_value)

            end_time = rospy.Time.now()

            episode_reward += r1
            s0 = s1

            begin_time = rospy.Time.now()
            agent.learn()
            end_time = rospy.Time.now()

            if done:
                break

        print(episode, ': ', episode_reward)

        x.append(episode)
        y.append(episode_reward)

        sio.savemat(os.path.dirname(os.path.realpath(__file__)) + '/step.mat',{'data': s},True,'5', False, False,'row')
        sio.savemat(os.path.dirname(os.path.realpath(__file__)) + '/reward.mat',{'data': r},True,'5', False, False,'row')
        sio.savemat(os.path.dirname(os.path.realpath(__file__)) + '/q_value.mat',{'data': q},True,'5', False, False,'row')
        sio.savemat(os.path.dirname(os.path.realpath(__file__)) + '/episode.mat',{'data': x},True,'5', False, False,'row')
        sio.savemat(os.path.dirname(os.path.realpath(__file__)) + '/total_reward.mat',{'data': y},True,'5', False, False,'row')

        # plot
        viz.line(
            y,
            x,
            win="gazebo1",
            name="line1",
            update=None,
            opts={
                'showlegend': True,
                'title': "reward-episode",
                'xlabel': "episode",
                'ylabel': "reward",
            },
        )
        viz.line(
            r,
            s,
            win="gazebo2",
            name="line2",
            update=None,
            opts={
                'showlegend': True,
                'title': "reward-step",
                'xlabel': "step",
                'ylabel': "reward",
            },
        )
        viz.line(
            q,
            s,
            win="gazebo3",
            name="line3",
            update=None,
            opts={
                'showlegend': True,
                'title': "q_value-step",
                'xlabel': "step",
                'ylabel': "Q value",
            },
        )

        # save model
        agent.save_data()

    rospy.spin()

        


