#! /usr/bin/env python
# coding :utf-8

from os import system
import rospy
from sensor_msgs.msg import LaserScan
from multi_rotor_avoidance_rl.msg import Acc
import time

last_time = time.time()
last_time_acc = time.time()
reboot_flag = False
episode = 0

def scanCB(msg):
    global last_time
    last_time = time.time()

def accCB(msg):
    global last_time_acc
    last_time_acc = time.time()

if __name__ == '__main__':

    rospy.init_node("reboot_node")
    scanSub = rospy.Subscriber("iris/scan", LaserScan, scanCB)
    accSub = rospy.Subscriber("/accel_body", Acc, accCB)

    while True:
        if (time.time() - last_time) > 20.0 or (time.time() - last_time_acc) > 20:
            if reboot_flag == False:
                print("Reboot in 10s.")
                system("shutdown -r -t 10")
                reboot_flag = True
        else:
            print("Not shutdown.")  

        time.sleep(1)      



    