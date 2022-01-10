#! /usr/bin/env python
#-*- coding: UTF- -*-

"""
使用visdom打印原始命令、命令和无人机状态
"""

# plot
import rospy
import visdom
import math
import numpy as np
from mavros_msgs.msg import PositionTarget
from geometry_msgs.msg import TwistStamped

myPT = PositionTarget()
myrawPT = PositionTarget()
myTS = TwistStamped()

viz = visdom.Visdom(env="formation")

opts1 = {
    'showlegend': True,
    'title': 'yaw_rate cmd/real',
    'xlable': 'time',
    'ylable': 'data',
    "legend":['cmd','real', 'raw']
}

opts2 = {
    'showlegend': True,
    'title': 'velocity_x cmd/real',
    'xlable': 'time',
    'ylable': 'data',
    "legend":['cmd','real', 'raw']
}


yaw_rate = []
velocity_x = []

def cmdCB(msg):
    global myPT
    myPT = msg

    # update queue
    if (len(yaw_rate) > 500):
        yaw_rate.pop(0)
        velocity_x.pop(0)

    yaw_rate.append([myPT.yaw_rate, myTS.twist.angular.z, myrawPT.yaw_rate])
    velocity_x.append([myPT.velocity.x, myTS.twist.linear.x, myrawPT.velocity.x])

    # plot
    x1 = range(0, len(yaw_rate))
    x2 = range(0, len(velocity_x))

    viz.line(yaw_rate, x1, win="formation1", update=None, opts=opts1)
    viz.line(velocity_x, x2, win="formation2", update=None, opts=opts2)

def twistCB(msg):
    global myTS
    myTS = msg

def rawCB(msg):
    global myrawPT
    myrawPT = msg

if __name__ == "__main__":
    
    rospy.init_node("plot")

    cmdSub = rospy.Subscriber("mod_cmd", PositionTarget, cmdCB, queue_size=10)
    rawSub = rospy.Subscriber("raw_cmd", PositionTarget, rawCB, queue_size=10)
    twistSub = rospy.Subscriber("iris_0/mavros/local_position/velocity_body", TwistStamped, twistCB, queue_size=10)

    rospy.spin()