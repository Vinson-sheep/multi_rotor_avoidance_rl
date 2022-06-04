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
import torch

import SAC_LSTM
import SAC_GRU

load_progress = False

if load_progress:
    load_progress = True
    load_buffer_flag = True
    load_actor_flag = True
    load_critic_flag = True
    load_log_alpha_flag = True
    load_optim_flag = True
else:
    load_progress = False
    load_buffer_flag = False
    load_actor_flag = False
    load_critic_flag = False
    load_log_alpha_flag = False
    load_optim_flag = False

fix_actor_flag = False

policy = "SAC_LSTM" # SAC_LSTM / SAC_GRU
game_name = "TRAIN" # EMPTY / TRAIN / TEST[1-4]

state_dim = 41
action_dim = 2
hidden_dim = 128

max_episode = 500
max_step_size = 300
init_episode = 0

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
    'load_buffer_flag': load_buffer_flag,
    'load_actor_flag': load_actor_flag,
    'load_critic_flag': load_critic_flag,
    'load_log_alpha_flag': load_log_alpha_flag,
    'load_optim_flag': load_optim_flag,
    'fix_actor_flag': fix_actor_flag,
}

if (policy == "SAC_LSTM"):
    agent = SAC_LSTM.SAC_LSTM(**kwargs)
if (policy == "SAC_GRU"):
    agent = SAC_GRU.SAC_GRU(**kwargs)

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
        np.save(url + "alpha_losses.npy", alpha_losses)
        np.save(url + "alphas.npy", alphas)
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

    alpha_losses = np.load(url + "alpha_losses.npy")
    alphas = np.load(url + "alphas.npy")

    for i in range(alpha_losses.size):  writer.add_scalar("Loss/alpha_loss", alpha_losses[i], global_step=i) 
    for i in range(alphas.size):  writer.add_scalar("Loss/alpha", alphas[i], global_step=i) 

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
            state = env.start()
            print("Game start!")
        else:
            state = env.reset()

        last_action = 2*np.random.rand(action_dim) - 1
        episode_state = []
        episode_action = []
        episode_last_action = []
        episode_reward = []
        episode_next_state = []
        episode_done = []        
        
        if (policy == "SAC_LSTM"):
            hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda(), \
                torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda())  # initialize hidden state for lstm, (hidden, cell), each is (batch, layer, dim) 
        if (policy == "SAC_GRU"):
            hidden_out = torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda()

        total_reward = 0
        episode_begin_time = rospy.Time.now()

        for step in range(max_step_size):

            hidden_in = hidden_out
            # choose action
            action, hidden_out = agent.act(state, last_action, hidden_in)

            if step == 0:
                ini_hidden_in = hidden_in
                ini_hidden_out = hidden_out


            step_begin_time = rospy.Time.now()

            # DEBUG
            pt = PositionTarget()
            pt.velocity.x = (action[0]+1)/4.0
            pt.yaw_rate = action[1]
            rawCmdPub.publish(pt)

            # DEBUG (useless in training)
            pt.velocity.x = (action[0]+1)/4.0
            pt.yaw_rate = action[1]
            modCmdPub.publish(pt)    

            # agent learn
            if episode < init_episode: agent.fix_actor_flag = True
            else: agent.fix_actor_flag = False
            
            learnThread().start()

            # step
            next_state, reward, done = env.step(step_time, (action[0]+1)/4.0, 0, action[1])        

            # # DEBUG
            msg = State()
            msg.header.stamp = rospy.Time.now()
            msg.cur_state = state
            msg.next_state = next_state
            statePub.publish(msg)

            episode_state.append(state)
            episode_action.append(action)
            episode_last_action.append(last_action)
            episode_reward.append(reward)
            episode_next_state.append(next_state)
            episode_done.append(done) 

            # plot and save
            step_rewards = np.append(step_rewards, reward)
            writer.add_scalar("Performance/step_reward", reward, global_step=step_rewards.size-1)  
            writer.add_scalar("DEBUG/step_time", (rospy.Time.now() - step_begin_time).to_sec(), global_step=step_rewards.size-1)  

            # other
            total_reward += reward
            state = next_state
            last_action = action

            if done: break
            if rospy.is_shutdown(): break

        agent.buffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action, \
            episode_reward, episode_next_state, episode_done)

        episode_time = (rospy.Time.now() - episode_begin_time).to_sec()
        episode_rewards = np.append(episode_rewards, total_reward)
        episode_times = np.append(episode_times, episode_time)
        writer.add_scalar("Performance/episode_reward", total_reward, global_step=episode_rewards.size-1)  
        writer.add_scalar("Performance/episode_time", episode_time, global_step=episode_times.size-1)  

        if rospy.is_shutdown(): break

        saveThread().start()

    rospy.spin()