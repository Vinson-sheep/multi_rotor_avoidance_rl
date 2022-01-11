#! /usr/bin/env python
#-*- coding: UTF- -*-

"""
 record data for variance calculating.
"""

# plot
import rospy
import os
import pickle
from mavros_msgs.msg import PositionTarget
from geometry_msgs.msg import TwistStamped

load_data = True

count = 0

twist = TwistStamped()

yaw_rate_cmd = []
velocity_x_cmd = []
yaw_rate_real = []
velocity_x_real = []

last_yaw_rate_cmd = 0
last_velocity_x_cmd = 0
last_yaw_rate_real = 0
last_velocity_x_real = 0

begin_time = None
has_reset = True

# file url
data_url = os.path.dirname(os.path.realpath(__file__)) + '/var.bin'

def cmdCB(msg):
    global count
    count += 1

    global begin_time
    begin_time = rospy.Time.now()

    global has_reset
    has_reset = False

    global last_yaw_rate_cmd
    global last_velocity_x_cmd
    global last_yaw_rate_real
    global last_velocity_x_real

    # append data
    yaw_rate_cmd.append(msg.yaw_rate - last_yaw_rate_cmd)
    velocity_x_cmd.append(msg.velocity.x - last_velocity_x_cmd)
    yaw_rate_real.append(twist.twist.angular.z - last_yaw_rate_real)
    velocity_x_real.append(twist.twist.linear.x - last_velocity_x_real)

    # save data
    if (count % 300 == 0):
        count = 0
        save_file = open(data_url,"wb")
        pickle.dump(yaw_rate_cmd, save_file)
        pickle.dump(velocity_x_cmd, save_file)
        pickle.dump(yaw_rate_real, save_file)
        pickle.dump(velocity_x_real, save_file)
        print("save data at length: %d" % len(yaw_rate_cmd))
        save_file.close()

    # update last data
    last_yaw_rate_cmd = msg.yaw_rate
    last_velocity_x_cmd = msg.velocity.x
    last_yaw_rate_real = twist.twist.angular.z
    last_velocity_x_real = twist.twist.linear.x


def realCB(msg):
    global twist
    twist = msg

def resetCB(event):
    global begin_time
    global has_reset

    if not has_reset and (rospy.Time.now() - begin_time) > rospy.Duration(2):

        global last_yaw_rate_cmd
        global last_velocity_x_cmd
        global last_yaw_rate_real
        global last_velocity_x_real
        last_yaw_rate_cmd = 0
        last_velocity_x_cmd = 0
        last_yaw_rate_real = 0
        last_velocity_x_real = 0

        
        has_reset = True

        print("reset.")

if __name__ == "__main__":
    
    rospy.init_node("var_collect")

    begin_time = rospy.Time.now()

    if load_data == True:
        load_file = open(data_url,"rb")
        yaw_rate_cmd = pickle.load(load_file)
        velocity_x_cmd = pickle.load(load_file)
        yaw_rate_real = pickle.load(load_file)
        velocity_x_real = pickle.load(load_file)
        print("load data.")

    cmdSub = rospy.Subscriber("/mod_cmd", PositionTarget, cmdCB, queue_size=10)
    realSub = rospy.Subscriber("/iris_0/mavros/local_position/velocity_body", TwistStamped, realCB, queue_size=10)
    resetTimer = rospy.Timer(rospy.Duration(2.0), resetCB)

    rospy.spin()