#! /usr/bin/env python
#-*- coding: UTF- -*-

# from pykalman import KalmanFilter
# from pykalman import UnscentedKalmanFilter

from pykalman import KalmanFilter
import numpy as np
import math

def is_valid(state, action, time_step):
    """
    determine whether action in state after time_step is valid
    return:
        - True if action is valid
        - valid reward
    dead reckoning
    """

    # restore range msg
    range_msg = np.array(state)
    angle_msg = np.array([ -2.3 + i*0.22 for i in range(len(range_msg))])
    # laser
    np.cos(angle_msg)
    laser_x = range_msg*np.cos(angle_msg)
    laser_y = range_msg*np.sin(angle_msg)
    
    radius = action[0]/action[1]
    theta = action[1]*time_step
    next_x = radius*math.sin(theta)
    next_y = radius - radius*math.cos(theta)


    # calculate 
    distance = (laser_x - next_x)**2 + (laser_y - next_y)**2

    valid_reward = 0
    flag = False
    
    return flag, valid_reward

is_valid([1,2,3,4,5,6], [0.5, -2.5], 0.5)