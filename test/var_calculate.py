#! /usr/bin/env python
#-*- coding: UTF- -*-

"""
    load data, and calculate variance.
"""

# plot
import os
import pickle
import numpy as np


# file url
data_url = os.path.dirname(os.path.realpath(__file__)) + '/var.bin'

if __name__ == "__main__":
    
    # load data
    load_file = open(data_url,"rb")
    yaw_rate_cmd = pickle.load(load_file)
    velocity_x_cmd = pickle.load(load_file)
    yaw_rate_real = pickle.load(load_file)
    velocity_x_real = pickle.load(load_file)

    # calculate variance
    print("The variance of yaw_rate_cmd is %f" % (np.var(yaw_rate_cmd)*1000))
    print("The variance of velocity_x_cmd is %f" % (np.var(velocity_x_cmd)*1000))
    print("The variance of yaw_rate_real is %f" % (np.var(yaw_rate_real)*1000))
    print("The variance of velocity_x_real is %f" % (np.var(velocity_x_real)*1000))


    
