#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

from common.game_training import Game
from mavros_msgs.msg import PositionTarget
import rospy
import DDPG
import TD3
import numpy as np

import visdom
import scipy.io as sio
import os
import threading
from multi_rotor_avoidance_rl.msg import State

# hyper parameter
epsilon = 0.8
epsilon_decay = 0.9999

load_able = False # True if you want to load previous data

policy = "TD3" # DDPG or TD3
filter = "NONE" # NONE / FIR / MAF / FOLF
game_name = "train_env_3m"

# Median Average Filter
window_size = 6
# First-Order Lag Filter
action_discount = 0.2

# DDPG and TD3 params
state_dim = 41
action_dim = 2
# hidden_dim = 500
hidden_dim = 300

discount = 0.99
# actor_lr = 1e-4
# critic_lr = 1e-2
actor_lr = 3e-5
critic_lr = 3e-4
tau = 0.01
buffer_size = 20000
batch_size = 512
alpha = 0.3
hyper_parameters_eps = 0.2

# td3 excluded
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2

# game params
load_buffer_flag = False
load_actor_flag = True
load_critic_flag = True
load_optim_flag = False
fix_actor_flag = True

max_episode = 30
max_step_size = 300

# variable
e = []
t = []
r = []
s = []
q = []
step_count_begin = 0
episode_begin = 0
agent = None

viz = visdom.Visdom(env="line")

# plot params
opts1={
    'showlegend': True,
    'title': "reward-episode",
    'xlabel': "episode",
    'ylabel': "reward",
}
opts2={
    'showlegend': True,
    'title': "reward-step",
    'xlabel': "step",
    'ylabel': "reward",
}
opts3={
    'showlegend': True,
    'title': "q_value-step",
    'xlabel': "step",
    'ylabel': "Q value",
}

# file url
data_url = os.path.dirname(os.path.realpath(__file__)) + '/data/' + policy + '/'

step_time = 0.1

class myThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        # save data
        sio.savemat(data_url + 'step.mat',{'data': s},True,'5', False, False,'row')
        sio.savemat(data_url + 'reward.mat',{'data': r},True,'5', False, False,'row')
        sio.savemat(data_url + 'q_value.mat',{'data': q},True,'5', False, False,'row')
        sio.savemat(data_url + 'episode.mat',{'data': e},True,'5', False, False,'row')
        sio.savemat(data_url + 'total_reward.mat',{'data': t},True,'5', False, False,'row')
        sio.savemat(data_url + 'epsilon.mat',{'data': epsilon},True,'5', False, False,'row')

        # plot
        viz.line(t, e, win="gazebo1", name="line1", update=None, opts=opts1)
        viz.line(r, s, win="gazebo2", name="line2:", update=None, opts=opts2)
        viz.line(q, s, win="gazebo3", name="line3", update=None, opts=opts3)

        # save model
        agent.save(data_url + policy)

        rospy.loginfo("save temperory variables, plot, and save models.")

if __name__ == '__main__':

    # initialize ros
    rospy.init_node("training_node")

    # raw data
    rawCmdPub = rospy.Publisher("raw_cmd", PositionTarget, queue_size=1)
    modCmdPub = rospy.Publisher("mod_cmd", PositionTarget, queue_size=1)
    statePub = rospy.Publisher("state", State, queue_size=1)

    # wait for world building
    rospy.sleep(rospy.Duration(3))

    # initialize environment
    env = Game("iris_0", game_name)

    # initialize agent
    kwargs = {
        'policy': policy,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': hidden_dim,
        'discount': discount,
        'actor_lr': actor_lr,
        'critic_lr': critic_lr,
        'tau': tau,
        'buffer_size': buffer_size,
        'batch_size': batch_size,
        'alpha': alpha,
        'hyper_parameters_eps': hyper_parameters_eps,
        'policy_noise': policy_noise,
        'noise_clip': noise_clip,
        'policy_freq': policy_freq,
        'load_buffer_flag': load_buffer_flag,
        'load_actor_flag': load_actor_flag,
        'load_critic_flag': load_critic_flag,
        'load_optim_flag': load_optim_flag,
        'fix_actor_flag': fix_actor_flag
    }

    if (policy == "DDPG"):
        agent = DDPG.Agent(**kwargs)
    if (policy == "TD3"):
        agent = TD3.Agent(**kwargs)
    
    # create mat file
    if not os.path.exists(data_url + 'episode.mat'):
        os.system(r"touch {}".format(data_url + 'episode.mat'))
    if not os.path.exists(data_url + 'total_reward.mat'):
        os.system(r"touch {}".format(data_url + 'total_reward.mat'))
    if not os.path.exists(data_url + 'reward.mat'):
        os.system(r"touch {}".format(data_url + 'reward.mat'))
    if not os.path.exists(data_url + 'step.mat'):
        os.system(r"touch {}".format(data_url + 'step.mat'))
    if not os.path.exists(data_url + 'q_value.mat'):
        os.system(r"touch {}".format(data_url + 'q_value.mat'))
    if not os.path.exists(data_url + 'epsilon.mat'):
        os.system(r"touch {}".format(data_url + 'epsilon.mat'))

    # load data if true
    if load_able == True:

        e = list(sio.loadmat(data_url + 'episode.mat')['data'][0])
        t = list(sio.loadmat(data_url + 'total_reward.mat')['data'][0])
        r = list(sio.loadmat(data_url + 'reward.mat')['data'][0])
        s = list(sio.loadmat(data_url + 'step.mat')['data'][0])
        q = list(sio.loadmat(data_url + 'q_value.mat')['data'][0])
        epsilon = list(sio.loadmat(data_url + 'epsilon.mat')['data'][0])[0]

        print("restore epsilon:", epsilon)

        if len(r) > 0:
            step_count_begin = s[-1]
            print("restore step count:", step_count_begin)
        if len(e) > 0:
            episode_begin = e[-1] + 1
            print("restore episode:", episode_begin)

    # start to train
    for episode in range(episode_begin, max_episode):

        if episode == episode_begin:
            s0 = env.start()
            print("start!")
        else:
            s0 = env.reset()
            print("reset.")

        episode_reward = 0

        # no filter
        if (filter == "NONE"):

            for step in range(max_step_size):

                step_count_begin += 1
                s.append(step_count_begin)

                # DEBUG
                msg = State()
                msg.header.stamp = rospy.Time.now()
                msg.data = s0
                statePub.publish(msg)

                a0 = agent.act(s0)

                # DEBUG
                pt = PositionTarget()
                pt.velocity.x = (a0[0]+1)/4.0
                pt.yaw_rate = a0[1]
                rawCmdPub.publish(pt)

                # E-greedy
                if epsilon > np.random.random():
                    a0[0] = np.clip(a0[0] + np.random.choice([-1, 1])* np.random.random()*0.8, -1.0, 1.0)
                    a0[1] = np.clip(a0[1] + np.random.choice([-1, 1])* np.random.random()*0.5, -1.0, 1.0)

                # DEBUG
                pt.velocity.x = (a0[0]+1)/4.0
                pt.yaw_rate = a0[1]
                modCmdPub.publish(pt)

                s1, r1, done = env.step(step_time, pt.velocity.x, 0, pt.yaw_rate)
                
                q_value = agent.put(s0, a0, r1, s1, done)

                epsilon = max(epsilon_decay*epsilon, 0.10)

                r.append(r1)
                q.append(q_value)

                episode_reward += r1
                s0 = s1

                agent.learn()

                if done:
                    break

        # FIR filter
        if (filter == "FIR"):
            pass

        # Median Average Filter
        if (filter == "MAF"):

            window_vx = [0.0]*window_size
            window_vyaw = [0.0]*window_size

            for step in range(max_step_size):

                step_count_begin += 1
                s.append(step_count_begin)

                # DEBUG
                msg = State()
                msg.header.stamp = rospy.Time.now()
                msg.data = s0
                statePub.publish(msg)

                a0 = agent.act(s0)

                # E-greedy
                if epsilon > np.random.random():
                    a0[0] = np.clip(a0[0] + np.random.choice([-1, 1])* np.random.random()*0.5, -1.0, 1.0)
                    a0[1] = np.clip(a0[1] + np.random.choice([-1, 1])* np.random.random()*0.5, -1.0, 1.0)

                # DEBUG
                pt = PositionTarget()
                pt.velocity.x = (a0[0]+1)/4.0
                pt.yaw_rate = a0[1]
                rawCmdPub.publish(pt)

                # update queue
                window_vx.pop(0)
                window_vx.append(a0[0])
                window_vyaw.pop(0)
                window_vyaw.append(a0[1])

                # copy data and sort
                window_vyaw_copy = window_vyaw[:]
                window_vyaw_copy.sort()
                window_vx_copy = window_vx[:]
                window_vx_copy.sort()

                # DEBUG
                if window_size <= 2:
                    pt.velocity.x = (np.mean(window_vx_copy)+1)/4.0
                    pt.yaw_rate = np.mean(window_vyaw_copy)
                else:
                    pt.velocity.x = (np.mean(window_vx_copy[1:-1])+1)/4.0
                    pt.yaw_rate = np.mean(window_vyaw_copy[1:-1])
                modCmdPub.publish(pt)

                s1, r1, done = env.step(step_time, pt.velocity.x, 0, pt.yaw_rate)
                
                q_value = agent.put(s0, a0, r1, s1, done)

                epsilon = max(epsilon_decay*epsilon, 0.10)

                r.append(r1)
                q.append(q_value)

                episode_reward += r1
                s0 = s1

                agent.learn()
                
                if done:
                    break

                

        # First-Order Lag Filter
        if (filter == "FOLF"):

            momentum = [0.0]*action_dim

            for step in range(max_step_size):

                step_count_begin += 1
                s.append(step_count_begin)

                # DEBUG
                msg = State()
                msg.header.stamp = rospy.Time.now()
                msg.data = s0
                statePub.publish(msg)

                a0 = agent.act(s0)

                # E-greedy
                if epsilon > np.random.random():
                    a0[0] = np.clip(a0[0] + np.random.choice([-1, 1])* np.random.random()*0.5, -1.0, 1.0)
                    a0[1] = np.clip(a0[1] + np.random.choice([-1, 1])* np.random.random()*0.5, -1.0, 1.0)

                # DEBUG
                pt = PositionTarget()
                pt.velocity.x = (a0[0]+1)/4.0
                pt.yaw_rate = a0[1]
                rawCmdPub.publish(pt)

                # monmentum
                momentum[0] = (1-action_discount)*momentum[0] + action_discount*a0[0]
                momentum[1] = (1-action_discount)*momentum[1] + action_discount*a0[1]

                # DEBUG
                pt.velocity.x = (momentum[0]+1)/4.0
                pt.yaw_rate = momentum[1]
                modCmdPub.publish(pt)

                s1, r1, done = env.step(step_time, pt.velocity.x, 0, pt.yaw_rate)
                
                q_value = agent.put(s0, a0, r1, s1, done)

                epsilon = max(epsilon_decay*epsilon, 0.10)

                r.append(r1)
                q.append(q_value)

                episode_reward += r1
                s0 = s1

                agent.learn()

                if done:
                    break

        print("eps = ", epsilon)
        print(episode, ': ', episode_reward)

        e.append(episode)
        t.append(episode_reward)

        myThread().start()

    rospy.spin()