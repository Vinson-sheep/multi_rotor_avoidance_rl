#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

import common.game_testing as game
import rospy
import numpy as np
from ddpg_brain import Agent
import scipy.io as sio
import os
import pickle

# hyper parameter
max_testing_num = 100
cur_testing_num = 0
cur_success_num = 0

restore_able = True
testing_num_begin = 0

params = {
    'gamma': 0.90,
    'actor_lr': 0.0001,
    'critic_lr': 0.0001,
    'tau': 0.01,
    'buffer_size': 100000,
    'batch_size': 512,
    'alpha': 0.3,
    'hyper_parameters_eps': 0.2,
    'load_data': True
}

def save():

    save_file = open(os.path.dirname(os.path.realpath(__file__)) + '/ddpg_data/temp_t.bin',"wb")
    pickle.dump(cur_testing_num,save_file)
    pickle.dump(cur_success_num,save_file)
    pickle.dump(max_testing_num,save_file)

    print("store cur_testing_num: ", cur_testing_num, "cur_success_num: ", cur_success_num, 
        "max_testing_num: ", max_testing_num)

    save_file.close()
    

def load():

    load_file = open(os.path.dirname(os.path.realpath(__file__)) + '/ddpg_data/temp_t.bin',"rb")
    cur_testing_num=pickle.load(load_file)
    cur_success_num=pickle.load(load_file)
    max_testing_num=pickle.load(load_file)

    print("restore cur_testing_num: ", cur_testing_num, "cur_success_num: ", cur_success_num, 
            "max_testing_num: ", max_testing_num)

    return cur_testing_num, cur_success_num, max_testing_num

if __name__ == '__main__':
    rospy.init_node("test")
    # global env
    env = game.Game("iris_0")

    # wait for world building
    rospy.sleep(rospy.Duration(3))

    agent1 = Agent(**params)

    # load data
    if restore_able == True:
        cur_testing_num, cur_success_num, max_testing_num = load()

    cur_testing_num += 1
    testing_num_begin = cur_testing_num


    for episode in range(testing_num_begin, max_testing_num):

        cur_testing_num = episode

        if episode == testing_num_begin:
            s0 = env.start()
            print("start testing！")
        else:
            s0 = env.reset()
            print("restore testing.")

        for step in range(500):
            
            a0 = agent.act(s0)

            s1, _, done = env.step(0.1, 0.5*a0[0], 0, a0[1])

            s0 = s1

            crash_indicator, _, _ = env.is_crashed()
                

            if done == True:
                if crash_indicator == True:
                    print("crashed!")
                else:
                    cur_success_num += 1
                    print("arrived")
                break

        print('[' + str(episode) + ']', ' success_rate:', float(cur_success_num)/float(max_testing_num) * 100, "%")

        save()


    rospy.spin()

        


