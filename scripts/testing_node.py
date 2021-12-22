#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

from common.game_testing import Game
import rospy
import DDPG
import TD3
import numpy as np
import os
import pickle

# hyper parameter
max_testing_num = 100

restore_able = False

policy = "DDPG" # DDPG or TD3
game_name = "corridor" # corridor / cluster

# DDPG and TD3 params
state_dim = 39
action_dim = 2

# variable
testing_num_begin = 0
cur_testing_num = 0
cur_success_num = 0

# file url
data_url = os.path.dirname(os.path.realpath(__file__)) + '/data/' + policy + '/'

def save():

    save_file = open(data_url + 'temp_test.bin',"wb")
    pickle.dump(cur_testing_num,save_file)
    pickle.dump(cur_success_num,save_file)
    pickle.dump(max_testing_num,save_file)

    print("store cur_testing_num: ", cur_testing_num, "cur_success_num: ", cur_success_num, 
        "max_testing_num: ", max_testing_num)

    save_file.close()
    

def load():

    load_file = open(data_url + 'temp_test.bin',"rb")
    cur_testing_num=pickle.load(load_file)
    cur_success_num=pickle.load(load_file)
    max_testing_num=pickle.load(load_file)

    print("restore cur_testing_num: ", cur_testing_num, "cur_success_num: ", cur_success_num, 
            "max_testing_num: ", max_testing_num)

    return cur_testing_num, cur_success_num, max_testing_num

if __name__ == '__main__':

    # initialize ros
    rospy.init_node("testing_node")

    # wait for world building
    rospy.sleep(rospy.Duration(3))
    
    # initialize environment
    env = Game("iris_0", game_name)

    # initialize agent

    kwargs = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'discount': 0,
        'actor_lr': 0,
        'critic_lr': 0,
        'tau': 0,
        'buffer_size': 0,
        'batch_size': 0,
        'alpha': 0,
        'hyper_parameters_eps': 0,
        'policy_noise': 0,
        'noise_clip': 0,
        'policy_freq': 0,
        'load_buffer_flag': False,
        'load_actor_flag': True,
        'load_critic_flag': False,
        'fix_actor_flag': True
    }

    if (policy == "DDPG"):
        agent = DDPG.Agent(**kwargs)
    if (policy == "TD3"):
        agent = TD3.Agent(**kwargs)

    # load data if true
    if restore_able == True:
        cur_testing_num, cur_success_num, max_testing_num = load()

    cur_testing_num += 1
    testing_num_begin = cur_testing_num

    # start to test
    for episode in range(testing_num_begin, max_testing_num):

        cur_testing_num = episode

        if episode == testing_num_begin:
            s0 = env.start()
            print("start testing！")
        else:
            s0 = env.reset()
            print("restore testing.")

        turning_flag = False # True if pose had been changed
        print("turning.")

        for step in range(500):
            
            # if need turning
            if (turning_flag == False):
                if (abs(s0[-1]) > 0.1):
                    s1, _, _ = env.step(0.1, 0, 0, s0[-1])
                    s0 = s1
                    continue
                else:
                    turning_flag = True
                    print("rl control")
            # rl control

            a0 = agent.act(s0)
            s1, _, done = env.step(0.1, (a0[0]+1)/4.0, 0, a0[1])
            
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

        


