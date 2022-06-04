#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

import rospy
from common.game import Game
from mavros_msgs.msg import PositionTarget
from multi_rotor_avoidance_rl.msg import State
import numpy as np
import os
import pickle
import torch

import SAC_LSTM
import SAC_GRU

load_progress = False

policy = "SAC_LSTM" # SAC_LSTM / SAC_GRU
filter = "NONE" # NONE / MAF / FOLF
game_name = "TEST1" # EMPTY / TRAIN / TEST[1-4]

# Median Average Filter
window_size = 4
# First-Order Lag Filter
action_discount = 0.5

state_dim = 41
action_dim = 2
hidden_dim = 128

max_episode = 500
max_step_size = 200

# variable
episode_begin = 0
success_num = 0
crash_num = 0
agent = None

# file url
url = os.path.dirname(os.path.realpath(__file__)) + '/data/'

step_time = 0.2

# initialize agent
kwargs = {
    'load_buffer_flag': False,
    'load_actor_flag': True,
    'load_critic_flag': False,
    'load_log_alpha_flag': False,
    'load_optim_flag': False,
    'fix_actor_flag': True,
    'use_priority': False
}


if (policy == "SAC_LSTM"):
    agent = SAC_LSTM.SAC_LSTM(**kwargs)
if (policy == "SAC_GRU"):
    agent = SAC_GRU.SAC_GRU(**kwargs)


def save(episode, success_num, crash_num):
    save_file = open(url + 'temp_test.bin',"wb")
    pickle.dump(episode,save_file)
    pickle.dump(success_num,save_file)
    pickle.dump(crash_num, save_file)
    save_file.close()
    

def load():
    load_file = open(url + 'temp_test.bin',"rb")
    episode=pickle.load(load_file)
    success_num=pickle.load(load_file)
    crash_num = pickle.load(load_file)
    print("Restore episode = %d, success num = %d, crash num = %d." % (episode+1, success_num, crash_num))
    return episode, success_num, crash_num

if __name__ == '__main__':

    # initialize ros
    rospy.init_node("testing_node")

    # raw data
    rawCmdPub = rospy.Publisher("raw_cmd", PositionTarget, queue_size=1)
    modCmdPub = rospy.Publisher("mod_cmd", PositionTarget, queue_size=1)
    statePub = rospy.Publisher("state", State, queue_size=1)

    # wait for world building
    rospy.sleep(rospy.Duration(3))
    
    # initialize environment
    env = Game("iris", game_name)

    # load data if true
    if load_progress: 
        episode_begin, success_num, crash_num = load()
        episode_begin += 1

    # start to test
    for episode in range(episode_begin, max_episode):

        print("=====================================")
        print("=========== Episode %d ===============" % (episode))
        print("=====================================")

        if episode == episode_begin:
            state = env.start()
            print("Start testing！")
        else:
            state = env.reset()

        # filter initialize
        if (filter == "NONE"):
            pass
        if (filter == "MAF"):
            window_vx = [0.0]*window_size
            window_vyaw = [0.0]*window_size
        if (filter == "FOLF"):
            momentum = [0.0]*action_dim

        if (policy == "SAC_LSTM"):
            hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda(), \
                torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda())  # initialize hidden state for lstm, (hidden, cell), each is (batch, layer, dim) 
        if (policy == "SAC_GRU"):
            hidden_out = torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda()

        last_action = 2*np.random.rand(action_dim) - 1

        for step in range(max_step_size):

            hidden_in = hidden_out
            # choose action
            action, hidden_out = agent.act(state, last_action, hidden_in)

            # DEBUG
            pt = PositionTarget()
            pt.velocity.x = (action[0]+1)/4.0
            pt.yaw_rate = action[1]
            rawCmdPub.publish(pt)

            if (filter == "NONE"):
                # DEBUG
                pass
                
            if (filter == "MAF"):
                # update queue
                window_vx.pop(0)
                window_vx.append(action[0])
                window_vyaw.pop(0)
                window_vyaw.append(action[1])
                # copy data and sort
                window_vyaw_copy = window_vyaw[:]
                window_vyaw_copy.sort()
                window_vx_copy = window_vx[:]
                window_vx_copy.sort()
                # DEBUG
                pt.velocity.x = (np.mean(window_vx_copy[1:-1])+1)/4.0
                pt.yaw_rate = np.mean(window_vyaw_copy[1:-1])

            if (filter == "FOLF"):
                momentum[0] = (1-action_discount)*momentum[0] + action_discount*action[0]
                momentum[1] = (1-action_discount)*momentum[1] + action_discount*action[1]
                # DEBUG
                pt.velocity.x = (momentum[0]+1)/4.0
                pt.yaw_rate = momentum[1]

            # DEBUG
            modCmdPub.publish(pt)

            # step
            next_state, _, done = env.step(step_time, pt.velocity.x, 0, pt.yaw_rate)

            # DEBUG
            msg = State()
            msg.header.stamp = rospy.Time.now()
            msg.cur_state = state
            msg.next_state = next_state
            statePub.publish(msg)

            state = next_state
            last_action = action

            # check result
            crash_indicator, _, _ = env.is_crashed()
            arrive_indicator = env.is_arrived()


            if done == True:
                if crash_indicator == True:
                    crash_num += 1
                    print("Crashed!")
                elif arrive_indicator == True:
                    success_num += 1
                    print("Arrived")
                break

            if rospy.is_shutdown(): break


        print('[' + str(episode+1) + '] success_num = %d, crash_num = %d' %(success_num, crash_num))

        if rospy.is_shutdown(): break

        save(episode, success_num, crash_num)

    rospy.spin()