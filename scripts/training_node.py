#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

import rospy
from common.game import Game
from mavros_msgs.msg import PositionTarget
from multi_rotor_avoidance_rl.msg import State
import numpy as np
import os
import threading
from tensorboardX import SummaryWriter

import DDPG
import TD3
import SAC

load_progress = False
load_buffer_flag = False
load_actor_flag = False
load_critic_flag = False
load_log_alpha_flag = False
load_optim_flag = False

fix_actor_flag = False
use_priority = True

policy = "SAC" # DDPG / TD3 / SAC
game_name = "train_env_7m"

epsilon = 0.8  # TD3
epsilon_decay = 0.99995 # TD3

state_dim = 41
action_dim = 2

max_episode = 500
max_step_size = 300
init_episode = 50

K = 1

# variable
episode_rewards = np.array([])
episode_times = np.array([])
step_rewards = np.array([])
actor_losses = np.array([])
critic_losses = np.array([])
alpha_losses = np.array([])
alphas = np.array([])
agent = None

url = os.path.dirname(os.path.realpath(__file__)) + '/data/'
writer = SummaryWriter(url + '../../log')

step_time = 0.2

# initialize agent
kwargs = {
    'state_dim': state_dim,
    'action_dim': action_dim,
    'load_buffer_flag': load_buffer_flag,
    'load_actor_flag': load_actor_flag,
    'load_critic_flag': load_critic_flag,
    'load_log_alpha_flag': load_log_alpha_flag,
    'load_optim_flag': load_optim_flag,
    'fix_actor_flag': fix_actor_flag,
    'use_priority': use_priority
}

if (policy == "TD3"):
    agent = TD3.TD3(**kwargs)
if (policy == "DDPG"):
    agent = DDPG.DDPG(**kwargs)
if (policy == "SAC"):
    agent = SAC.SAC(**kwargs)

class saveThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):

        print("*Saving. Please don't close the window!")

        begin_time = rospy.Time.now()
        # save data
        np.save(url + "episode_rewards.npy", episode_rewards)
        np.save(url + "episode_times.npy", episode_times)
        np.save(url + "step_rewards.npy", step_rewards)
        np.save(url + "actor_losses.npy", actor_losses)
        np.save(url + "critic_losses.npy", critic_losses)
        if policy == "SAC": np.save(url + "alpha_losses.npy", alpha_losses)
        if policy == "SAC": np.save(url + "alphas.npy", alphas)
        if policy == "TD3" or policy == "DDPG": np.save(url + "epsilon.npy", epsilon)
        # save model
        agent.save()
        # print
        save_time = (rospy.Time.now() - begin_time).to_sec()
        writer.add_scalar("DEBUG/save_time", save_time, global_step=episode_rewards.size-1)  
        
        print("Saved. Time consumed = %f seconds." % (save_time))


class learnThread(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):

        global actor_losses
        global critic_losses
        global alpha_losses
        global alphas

        # agent learn
        begin_time = rospy.Time.now()
        for i in range(K): agent.update()
        learn_time = (rospy.Time.now() - begin_time).to_sec()
        # log
        actor_losses = np.append(actor_losses, agent.actor_loss)
        writer.add_scalar("Loss/actor_loss", agent.actor_loss, global_step=actor_losses.size-1) 
        critic_losses = np.append(critic_losses, agent.critic_loss)
        writer.add_scalar("Loss/critic_loss", agent.critic_loss, global_step=critic_losses.size-1) 
        if policy == "SAC":
            alpha_losses = np.append(alpha_losses, agent.alpha_loss)
            writer.add_scalar("Loss/alpha_loss", agent.alpha_loss, global_step=alpha_losses.size-1) 
            alphas = np.append(alphas, agent.alpha.item())
            writer.add_scalar("Loss/alpha", agent.alpha.item(), global_step=alphas.size-1) 

        if step_rewards.size % 100 == 0:
            print("Learned. Time consumed = %f seconds." % (learn_time))


def loadData():

    global episode_rewards
    global episode_times
    global step_rewards
    global actor_losses
    global critic_losses
    global alpha_losses
    global alphas
    global epsilon
    episode_rewards = np.load(url + "episode_rewards.npy")
    episode_times = np.load(url + "episode_times.npy")
    step_rewards = np.load(url + "step_rewards.npy")
    actor_losses = np.load(url + "actor_losses.npy")
    critic_losses = np.load(url + "critic_losses.npy")

    for i in range(episode_rewards.size): writer.add_scalar("Performance/episode_reward", episode_rewards[i], global_step=i)  
    for i in range(episode_times.size): writer.add_scalar("Performance/episode_time", episode_times[i], global_step=i)  
    for i in range(step_rewards.size): writer.add_scalar("Performance/step_reward", step_rewards[i], global_step=i)  
    for i in range(actor_losses.size): writer.add_scalar("Loss/actor_loss", actor_losses[i], global_step=i)  
    for i in range(critic_losses.size): writer.add_scalar("Loss/critic_loss", critic_losses[i], global_step=i)  

    if policy == "SAC": 
        alpha_losses = np.load(url + "alpha_losses.npy")
        alphas = np.load(url + "alphas.npy")

        for i in range(alpha_losses.size):  writer.add_scalar("Loss/alpha_loss", alpha_losses[i], global_step=i) 
        for i in range(alphas.size):  writer.add_scalar("Loss/alpha", alphas[i], global_step=i) 

    if policy == "TD3" or policy == "DDPG": epsilon = np.load(url + "epsilon.npy")

    print("1. Restore episode: %d" % (episode_rewards.size))
    print("2. Restore step: %d" % (step_rewards.size))
    


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
    env = Game("iris", game_name)
    
    # load data if true
    if load_progress: loadData()

    episode_begin = episode_rewards.size

    # start to train
    for episode in range(episode_begin, max_episode):

        print("=====================================")
        print("=========== Episode %d ===============" % (episode))
        print("=====================================")


        if episode == episode_begin:
            s0 = env.start()
            print("Game start!")
        else:
            s0 = env.reset()

        episode_reward = 0
        episode_begin_time = rospy.Time.now()

        for step in range(max_step_size):

            step_begin_time = rospy.Time.now()

            # choose action
            a0 = agent.act(s0)

            # DEBUG
            pt = PositionTarget()
            pt.velocity.x = (a0[0]+1)/4.0
            pt.yaw_rate = a0[1]
            rawCmdPub.publish(pt)

            if (policy == "TD3" or policy == "DDPG"):
                if epsilon > np.random.random():
                    a0 = (a0 + np.random.normal(0, 0.3, size=a0.size)).clip(-1.0, 1.0)

            # DEBUG
            pt.velocity.x = (a0[0]+1)/4.0
            pt.yaw_rate = a0[1]
            modCmdPub.publish(pt)    

            # agent learn
            if episode < init_episode: agent.fix_actor_flag = True
            else: agent.fix_actor_flag = False
            
            learnThread().start()

            # step
            s1, r1, done = env.step(step_time, pt.velocity.x, 0, pt.yaw_rate)        

            # DEBUG
            msg = State()
            msg.header.stamp = rospy.Time.now()
            msg.cur_state = s0
            msg.next_state = s1
            statePub.publish(msg)

            # save transition
            agent.put(s0, a0, r1, s1, done)

            # plot and save
            step_rewards = np.append(step_rewards, r1)
            writer.add_scalar("Performance/step_reward", r1, global_step=step_rewards.size-1)  
            writer.add_scalar("DEBUG/step_time", (rospy.Time.now() - step_begin_time).to_sec(), global_step=step_rewards.size-1)  

            # other
            epsilon = max(epsilon_decay*epsilon, 0.20)
            episode_reward += r1
            s0 = s1

            if done: break
            if rospy.is_shutdown(): break

        episode_time = (rospy.Time.now() - episode_begin_time).to_sec()
        episode_rewards = np.append(episode_rewards, episode_reward)
        episode_times = np.append(episode_times, episode_time)
        writer.add_scalar("Performance/episode_reward", episode_reward, global_step=episode_rewards.size-1)  
        writer.add_scalar("Performance/episode_time", episode_time, global_step=episode_times.size-1)  

        if policy == "DDPG" or policy == "TD3":
            print("epsilon = %f" % (epsilon))

        if rospy.is_shutdown(): break

        saveThread().start()

    rospy.spin()
