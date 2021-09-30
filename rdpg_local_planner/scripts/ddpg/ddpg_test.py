#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

import game
import rospy
import numpy as np
from ddpg import Agent
import visdom

epsilon = 0.9
epsilon_decay = 0.9999

def limit(x, max_x, min_x):
    return max(min(x, max_x), min_x)

if __name__ == '__main__':
    rospy.init_node("test")
    env = game.Game("iris_0")
    viz = visdom.Visdom(env="line")

    params = {
        'env': env,
        'gamma': 0.99,
        'actor_lr': 0.001,
        'critic_lr': 0.001,
        'tau': 0.02,
        'capacity': 10000,
        'batch_size': 32,
    }

    agent = Agent(**params)

    x = []
    y = []

    for episode in range(1000):
        if episode == 0:
            s0 = env.start()
            print("start!")
        else:
            s0 = env.reset()
            print("reset.")

        episode_reward = 0

        for step in range(500):
            a0 = agent.act(s0)

            # print("raw action: ", a0)

            if epsilon > np.random.random():
                a0[0] += np.random.random()*0.4
                a0[1] += (np.random.random()-0.5)*0.5
                a0[2] += (np.random.random()-0.5)*0.4

                a0[0] = limit(a0[0], 1.0, 0.0)
                a0[1] = limit(a0[1], 1.0, -1.0)
                a0[2] = limit(a0[2], 1.0, -1.0)
            epsilon *= epsilon_decay

            # print("cut action: ", a0)
            

            print("eps = ", epsilon)



            begin_time = rospy.Time.now()
            s1, r1, done = env.step(0.1, a0[0], a0[1], a0[2])
            agent.put(s0, a0, r1, s1, done)
            end_time = rospy.Time.now()

            # print("step time:", (end_time - begin_time).to_sec)

            

            episode_reward += r1
            s0 = s1

            begin_time = rospy.Time.now()
            agent.learn()
            end_time = rospy.Time.now()
            # print("learn time:", (end_time - begin_time).to_sec)

            if done:
                break

        print(episode, ': ', episode_reward)

        x.append(episode)
        y.append(episode_reward)

        # print(x)
        # print(y)

        viz.line(
            y,
            x,
            win="line1",
            name="line",
            update=None,
            opts={
                'showlegend': True,
                'title': "Demo line in Visdom",
                'xlabel': "episode",
                'ylabel': "reward",
            },
        )


        


