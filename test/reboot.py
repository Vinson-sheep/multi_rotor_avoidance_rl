#! /usr/bin/env python
# coding :utf-8

from os import system
import rospy
from sensor_msgs.msg import LaserScan
import time

last_time = time.time()
reboot_flag = False
episode = 0

def scanCB(msg):
    global last_time
    last_time = time.time()

if __name__ == '__main__':

    rospy.init_node("reboot_node")
    scanSub = rospy.Subscriber("iris/scan", LaserScan, scanCB)

    while True:
        if (time.time() - last_time) > 20.0:
            if reboot_flag == False:
                print("Reboot in 10s.")
                system("shutdown -r -t 10")
                reboot_flag = True
        else:
            print("Not shutdown.")  

        time.sleep(1)      



    