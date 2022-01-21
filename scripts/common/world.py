#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

import math

from torch import set_flush_denormal
import rospy
import random
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, DeleteModel,DeleteModelRequest
from tf.transformations import quaternion_from_euler
import os

class World:

    def __init__(self, safe_space, safe_radius, wall_rate = 1.0, cylinder_num = 100):
        """
            safe_space:
            safe_radius:
            wall_rate: thr propotion of having wall.
            cylidar_num:
        """

        # constant
        self.safe_space = safe_space
        self.safe_radius = safe_radius
        self.wall_rate = wall_rate
        self.cylinder_num = cylinder_num

        self.target_height = 1.0
        self.target_model_name = "target_ball"
        self.expend_radius = 3

        self.orientation_error = 0.2
        self.position_error = 1.0

        # initialize wall list
        self.wall_list = []

        # initialize cylinder list
        self.cylinder_list = []

        # wall param
        self.init_wall_param(safe_space, safe_radius)

        # service client
        self.spawnModelSdfClient = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        self.deleteModelClient = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)

        # XML
        self.url = os.path.dirname(os.path.realpath(__file__))

        f = open(self.url + '/ball/model.sdf')
        self.ballXML = f.read()
        f.close()

        f = open(self.url + '/cylinder/model.sdf')
        self.cylinderXML = f.read()
        f.close()

        f = open(self.url + '/wall/model.sdf')
        self.wallXML = f.read()
        f.close()

        # wait for service
        self.spawnModelSdfClient.wait_for_service()
        self.deleteModelClient.wait_for_service()


    def reset(self):
        """
            reset world
        """
    
        # reset wall
        if random.uniform(0.0, 1.0) < self.wall_rate:

            msg = SpawnModelRequest()
            msg.model_xml = self.wallXML
            msg.initial_pose.position.z = 0

            # reset upper wall
            msg.model_name = "wall_upper"
            self.wall_list.append(msg.model_name)
            msg.initial_pose.position.x = random.uniform(self.top_limit, self.top_limit + self.expend_radius)
            msg.initial_pose.position.y = random.uniform(-self.position_error, self.position_error)
            q = quaternion_from_euler(0, 0, math.pi/2 + random.uniform(-self.orientation_error, self.orientation_error))
            msg.initial_pose.orientation.x = q[0]
            msg.initial_pose.orientation.y = q[1]
            msg.initial_pose.orientation.z = q[2]
            msg.initial_pose.orientation.w = q[3]
            try:
                self.spawnModelSdfClient.call(msg)
            except:
                pass

            # reset left wall
            msg.model_name = "wall_left"
            self.wall_list.append(msg.model_name)
            msg.initial_pose.position.x = random.uniform(-self.position_error, self.position_error)
            msg.initial_pose.position.y = random.uniform(self.left_limit, self.left_limit + 2*self.expend_radius)
            q = quaternion_from_euler(0, 0, random.uniform(-self.orientation_error, self.orientation_error))
            msg.initial_pose.orientation.x = q[0]
            msg.initial_pose.orientation.y = q[1]
            msg.initial_pose.orientation.z = q[2]
            msg.initial_pose.orientation.w = q[3]
            try:
                self.spawnModelSdfClient.call(msg)
            except:
                pass

            # reset right wall
            msg.model_name = "wall_right"
            self.wall_list.append(msg.model_name)
            msg.initial_pose.position.x = random.uniform(-self.position_error, self.position_error)
            msg.initial_pose.position.y = random.uniform(self.right_limit - 2*self.expend_radius, self.right_limit)
            q = quaternion_from_euler(0, 0, random.uniform(-self.orientation_error, self.orientation_error))
            msg.initial_pose.orientation.x = q[0]
            msg.initial_pose.orientation.y = q[1]
            msg.initial_pose.orientation.z = q[2]
            msg.initial_pose.orientation.w = q[3]
            try:
                self.spawnModelSdfClient.call(msg)
            except:
                pass

            # reset bottom wall
            msg.model_name = "wall_bottom"
            self.wall_list.append(msg.model_name)
            msg.initial_pose.position.x = random.uniform(self.bottom_limit - 2*self.expend_radius, self.bottom_limit)
            msg.initial_pose.position.y = random.uniform(-self.position_error, self.position_error)
            q = quaternion_from_euler(0, 0, math.pi/2 + random.uniform(-self.orientation_error, self.orientation_error))
            msg.initial_pose.orientation.x = q[0]
            msg.initial_pose.orientation.y = q[1]
            msg.initial_pose.orientation.z = q[2]
            msg.initial_pose.orientation.w = q[3]
            try:
                self.spawnModelSdfClient.call(msg)
            except:
                pass

        
        # reset cylinder
        msg = SpawnModelRequest()
        msg.model_xml = self.cylinderXML
        msg.initial_pose.position.z = 1.5
        msg.initial_pose.orientation.x = 0
        msg.initial_pose.orientation.y = 0
        msg.initial_pose.orientation.z = 0
        msg.initial_pose.orientation.w = 1

        for i in range(0, int(self.cylinder_num)):
            msg.model_name = "cylinder_" + str(i)
            self.cylinder_list.append(msg.model_name)
            msg.initial_pose.position.x = random.uniform(-8, 8)
            msg.initial_pose.position.y = random.uniform(-8, 8)
            while not self.check_safe(msg.initial_pose.position.x, msg.initial_pose.position.y):
                msg.initial_pose.position.x = random.uniform(-8, 8)
                msg.initial_pose.position.y = random.uniform(-8, 8)
            try:
                self.spawnModelSdfClient.call(msg)
            except:
                pass

    def clear(self):
        """
            clear world
        """
        msg = DeleteModelRequest()
        # call service

        try:
            # delete wall
            for item in self.wall_list:
                msg.model_name = item
                self.deleteModelClient.call(msg)
            self.wall_list = []
            # delete cylinder
            for item in self.cylinder_list:
                msg.model_name = item
                self.deleteModelClient.call(msg)
            self.cylinder_list = []
            # delete ball
            msg.model_name = self.target_model_name
            self.deleteModelClient.call(msg)
        except:
            pass


    def set_target(self, x, y):
        """
            reset target ball
        """
        msg = SpawnModelRequest()
        msg.model_xml = self.ballXML
        msg.initial_pose.position.z = 0.5
        msg.initial_pose.orientation.x = 0
        msg.initial_pose.orientation.y = 0
        msg.initial_pose.orientation.z = 0
        msg.initial_pose.orientation.w = 1

        msg.model_name = self.target_model_name
        msg.initial_pose.position.x = x
        msg.initial_pose.position.y = y

        try:
            self.spawnModelSdfClient.call(msg)
        except:
            pass
    
    # tool function
    def set_wall_rate(self, wall_rate):
        self.wall_rate = wall_rate

    def set_cylinder_rate(self, cylinder_rate):
        self.cylinder_rate = cylinder_rate

    def init_wall_param(self, safe_space, safe_radius):
        self.top_limit = -100
        self.left_limit = -100
        self.right_limit = 100
        self.bottom_limit = 100
        top_i = -1
        left_i = -1
        right_i = -1
        bottom_i = -1
        for i in range(0, len(safe_space)):
            if safe_space[i][0] > self.top_limit:
                self.top_limit = safe_space[i][0]
                top_i = i
            if safe_space[i][1] > self.left_limit:
                self.left_limit = safe_space[i][1]
                left_i = i
            if safe_space[i][1] < self.right_limit:
                self.right_limit = safe_space[i][1]
                right_i = i
            if safe_space[i][0] < self.bottom_limit:
                self.bottom_limit = safe_space[i][0]
                bottom_i = i
        self.top_limit += safe_radius[top_i]
        self.left_limit += safe_radius[left_i]
        self.right_limit -= safe_radius[right_i]
        self.bottom_limit -= safe_radius[bottom_i]

    def check_safe(self, x, y):
        """
            check whether (x, y) falls into safe space.
            return:
                True - if not fall into
        """
        for i in range(0, len(self.safe_space)):
            distance_power = (self.safe_space[i][0] - x)**2 + (self.safe_space[i][1] - y)**2
            if distance_power < self.safe_radius[i]**2:
                return False
        return True


if __name__ == '__main__':

    rospy.init_node("test")
    
    world = World([[0, 0], [3, 0]], [1, 1], 1, 100)
    
    world.reset()
    world.set_target(1, 2)
    rospy.sleep(rospy.Duration(5))
    world.clear()

    rospy.spin()