#! /usr/bin/env python
#-*- coding: UTF- -*-


# plot
import rospy
import numpy as np
from multi_rotor_avoidance_rl.msg import State
from embedding_buffer import EmbeddingBuffer

load_able = True

buffer = EmbeddingBuffer()

if load_able:
    buffer.load()
    print("Restore buffer. size = %d" %(buffer.size()))

count = 0

def stateCB(msg):

    global buffer    
    global count
    count +=1

    # add data
    if buffer.size() < 100000:
        buffer.add(msg.cur_state, msg.next_state)

    # save data
    if count % 100 == 0:
        buffer.save()
        rospy.loginfo("Save data: %d" % (buffer.size()))


if __name__ == "__main__":
    
    rospy.init_node("test")
    rospy.loginfo("Start data collecting.")
    stateSub = rospy.Subscriber("/state", State, stateCB, queue_size=10)
    rospy.spin()