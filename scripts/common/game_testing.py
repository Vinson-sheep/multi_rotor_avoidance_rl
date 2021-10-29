#! /usr/bin/env python
#-*- coding: UTF-8 -*- 
from __future__ import print_function 
from math import atan2
import rospy
from rospy.client import spin
from rospy.rostime import Duration

# ros include
from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose, PoseStamped, TwistStamped, TwistWithCovarianceStamped
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, CommandBoolResponse
from mavros_msgs.srv import SetMode, SetModeRequest, SetModeResponse
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetModelState, GetModelState, GetModelStateRequest, GetModelStateResponse
from gazebo_msgs.msg import ModelState 
from tf.transformations import euler_from_quaternion, quaternion_from_euler

import random
import numpy as np
import math
import angles

class Game:

    def __init__(self, model_name):
        """
            initialize
        """


        self.model_name = model_name # uav name
        self.mavrosState = State()
        self.pose = Pose()
        self.twist = Twist()
        self.scan = LaserScan()
        self.body_v = TwistStamped()

        self.crash_limit = 0.30

        self.start_flag = False

        self.target_x = 10
        self.target_y = 10

        self.state_num = 24+4
        self.action_num = 2

        self.height = 3.0 # height of taking off

        self.target_distance = 7

        self.rate = rospy.Rate(20)

        self.hold_flag = False # if True, send hold command
        self.hold_able = False
        self.hold_pose = Pose()
        self.last_cmd_time = rospy.Time.now()

        # subscriber
        self.mavrosStateSub = rospy.Subscriber(self.model_name + "/mavros/state", State, self._mavrosStateCB)
        self.scanSub = rospy.Subscriber(self.model_name + "/scan_downsampled", LaserScan, self._scanCB)
        self.bodyVelocitySub = rospy.Subscriber(self.model_name + "/mavros/local_position/velocity_body", TwistStamped, self._bodyVelocityCB)



        # publisher
        self.ctrPub = rospy.Publisher(self.model_name + "/mavros/setpoint_raw/local", PositionTarget, queue_size=1)
        self.visionPub = rospy.Publisher(self.model_name + "/mavros/vision_pose/pose", PoseStamped, queue_size=1)
        self.visionVPub = rospy.Publisher(self.model_name + "/mavros/vision_speed/speed_twist_cov", TwistWithCovarianceStamped, queue_size=1)


        # service client
        self.setModelStateClient = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.armingClient = rospy.ServiceProxy(self.model_name + "/mavros/cmd/arming", CommandBool)
        self.setModeClient = rospy.ServiceProxy(self.model_name + "/mavros/set_mode", SetMode)
        self.modelStateClient = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self.resetWorldClient = rospy.ServiceProxy("/gazebo/reset_world", Empty)

        # Timer
        self.holdTimer = rospy.Timer(rospy.Duration(0.05), self._hold)
        self.visionTimer = rospy.Timer(rospy.Duration(0.02), self._vision)

    def reset(self):
        """
            TODO
        """

        # stop in place
        print("stoping.")
        # while not self._is_hold():
        for i in range(20):
            self._send_velocity_cmd(0, 0, 0)
            self.hold_flag = False
            self.rate.sleep()

        # go backward with low velocty
        print("go backward.")
        self.flag, _, self.crash_index = self.is_crashed()
        while self.flag == True:
            # calculate the vx and vy
            alpha = self.scan.angle_min + self.scan.angle_increment * self.crash_index

            vx = -math.cos(alpha) * 0.3
            vy = -math.sin(alpha) * 0.3 
            self._send_velocity_cmd(vx, vy, 0)
            self.hold_flag = False
            self.rate.sleep()
            self.flag, _, self.crash_index = self.is_crashed()

            

        # stop in place
        while not self._is_hold():
            self._send_velocity_cmd(0, 0, 0)
            self.hold_flag = False
            self.rate.sleep()


        # make sure uav is holding
        self.hold_able = True

        while not self._is_hold():
            rospy.sleep(rospy.Duration(0.1))

        # uav fly height and hold
        cur_pose = self.pose
        
        cmd_x = cur_pose.position.x
        cmd_y = cur_pose.position.y
        (_, _, cmd_yaw) = euler_from_quaternion([
                                                cur_pose.orientation.x,
                                                cur_pose.orientation.y,
                                                cur_pose.orientation.z,
                                                cur_pose.orientation.w
                                                ])

        while self._is_arrived(cmd_x, cmd_y, 6) == False or self._is_hold() == False:
            self._send_position_cmd(cmd_x, cmd_y, cmd_yaw, height=6)
            self.hold_flag = False
            self.rate.sleep()
        
        rospy.sleep(rospy.Duration(1))
        
        # fly home
        home_x = random.choice([-1, 1])* np.random.random()
        home_y = random.choice([-1, 1])* np.random.random()
        home_yaw = 3.1415926*random.choice([-1, 1])* np.random.random()

        while self._is_arrived(home_x, home_y, 6) == False or self._is_hold() == False:
            
            if home_x > self.pose.position.x:
                x_t = min(home_x, self.pose.position.x + 1)
            else:
                x_t = max(home_x, self.pose.position.x - 1)

            if home_y > self.pose.position.y:
                y_t = min(home_y, self.pose.position.y + 1)
            else:
                y_t = max(home_y, self.pose.position.y - 1)

            self._send_position_cmd(x_t, y_t, cmd_yaw, height=6)
            self.hold_flag = False
            self.rate.sleep()

        rospy.sleep(rospy.Duration(1))
        
        # flg down and hold
        while self._is_arrived(home_x, home_y, self.height) == False or self._is_hold() == False:
            self._send_position_cmd(home_x, home_y, home_yaw, height=self.height)
            self.hold_flag = False
            self.rate.sleep()

        self.hold_able = True

        rospy.sleep(rospy.Duration(1))

        rospy.loginfo("initialize uav position.")

        # randomize target point
        self.target_x = random.choice([self.target_distance, -self.target_distance]) + random.choice([-1, 1])* np.random.random()
        self.target_y = random.choice([self.target_distance, -self.target_distance]) + random.choice([-1, 1])* np.random.random()

        target_msg = ModelState()
        target_msg.model_name = 'unit_sphere'
        target_msg.pose.position.x = self.target_x
        target_msg.pose.position.y = self.target_y
        target_msg.pose.position.z = self.height - 1
        target_msg.pose.orientation.x = 0
        target_msg.pose.orientation.y = 0
        target_msg.pose.orientation.z = 0
        target_msg.pose.orientation.w = 1

        # call service
        self.setModelStateClient.wait_for_service()

        self.setModelStateClient.call(target_msg)
        rospy.loginfo("initialize target position.")

        return self._cur_state()

    def start(self):
        """
            send take-off command
        """
        

        # randomize target point
        self.target_x = random.choice([self.target_distance, -self.target_distance]) + random.choice([-1, 1])* np.random.random()
        self.target_y = random.choice([self.target_distance, -self.target_distance]) + random.choice([-1, 1])* np.random.random()

        target_msg = ModelState()
        target_msg.model_name = 'unit_sphere'
        target_msg.pose.position.x = self.target_x
        target_msg.pose.position.y = self.target_y
        target_msg.pose.position.z = self.height - 1
        target_msg.pose.orientation.x = 0
        target_msg.pose.orientation.y = 0
        target_msg.pose.orientation.z = 0
        target_msg.pose.orientation.w = 1

        # call service
        self.setModelStateClient.wait_for_service()

        self.setModelStateClient.call(target_msg)
        rospy.loginfo("initialize target position.")

        self.hold_able = False

        # wait for pose update
        rospy.sleep(rospy.Duration(1))

        # record current pose
        cur_pose = self.pose
        
        # prepare take-off data
        cmd_x = cur_pose.position.x
        cmd_y = cur_pose.position.y
        (_, _, cmd_yaw) = euler_from_quaternion([
                                                cur_pose.orientation.x,
                                                cur_pose.orientation.y,
                                                cur_pose.orientation.z,
                                                cur_pose.orientation.w
                                                ])

        # send a list of messages before taking off
        for i in range(100):
            self._send_position_cmd(cmd_x, cmd_y, cmd_yaw, height=self.height)
            self._send_position_cmd(1, 1, cmd_yaw, height=self.height)
            self.hold_flag = False

        # send arming command
        arm_cmd = CommandBoolRequest()
        arm_cmd.value = True
        self.armingClient.wait_for_service()

        count = 0
        begin_time = rospy.Time.now()
        while self.armingClient.call(arm_cmd).success == False and count < 10:
            rospy.sleep(rospy.Duration(1))
            count += 1

        if count<10:
            rospy.loginfo(self.model_name + ": arming.")
        else:
            rospy.logerr(self.model_name + ": arming fail.")
            return False
        
        # arming for 3 seconds
        rospy.sleep(rospy.Duration(3))

        # set offboard mode
        mode_cmd = SetModeRequest()
        mode_cmd.custom_mode = "OFFBOARD"
        resp = self.setModeClient.call(mode_cmd)
        
        if resp.mode_sent:
            rospy.loginfo(self.model_name + ": offboard command sent.")
        else:
            rospy.logerr(self.model_name + ": fail to send offboard command.")
            return False

        # continue to send message
        for i in range(200):
            self._send_position_cmd(cmd_x, cmd_y, cmd_yaw, height=self.height)
            self.hold_flag = False

        rospy.loginfo(self.model_name + ": take-off successfully. hold now.")

        # hold timer is able
        self.hold_able = True

        self.start_flag = True

        return self._cur_state()


    def is_crashed(self):
        """
            determine if the uav is crashed.
            return:
                - True if it was crashed.
                - crash reward
                - relative laser index (to check the direction)
        """
        self.laser_crashed_flag = False
        self.laser_crashed_reward = 0
        self.crash_index = -1

        for i in range(len(self.scan.ranges)):
            if self.scan.ranges[i] < 2*self.crash_limit:
                self.laser_crashed_reward = - 40.0
            if self.scan.ranges[i] < self.crash_limit:
                self.laser_crashed_reward = - 200.0
                self.laser_crashed_flag = True
                self.crash_index = i
                break
        return self.laser_crashed_flag, self.laser_crashed_reward, self.crash_index


    def step(self, time_step=0.1, vx=0.1, vy=0.1, yaw_rate=0.1):
        """
            game step
        """
        self.hold_able = False
        # record last x and y
        last_pos_x_uav = self.pose.position.x
        last_pos_y_uav = self.pose.position.y
        last_distance = math.sqrt((self.target_x - last_pos_x_uav)**2 + (self.target_y - last_pos_y_uav)**2)

        # send control command for time_step period
        time = rospy.Time.now()
        while (rospy.Time.now() - time) < Duration(time_step):

            self._send_velocity_cmd(vx, vy, yaw_rate)
            self.hold_flag = False

            crash_indicator, _, _ = self.is_crashed()
            if crash_indicator == True:
                break

        cur_pos_x_uav = self.pose.position.x
        cur_pos_y_uav = self.pose.position.y
        cur_distance = math.sqrt((self.target_x - cur_pos_x_uav)**2 + (self.target_y - cur_pos_y_uav)**2)
        
        # distance reward
        distance_reward = last_distance - cur_distance

        self.done = False

        # arrive reward
        self.arrive_reward = 0
        if cur_distance < 0.3:
            self.arrive_reward = 100
            self.done = True

        # crash reward
        crash_indicator, crash_reward, _ = self.is_crashed()
        if crash_indicator == True:
            self.done = True

        # laser reward
        state = np.array(self.scan.ranges) / float(self.scan.range_max)
        # print(state)
        laser_reward = sum(state) - len(state)

        # linear punish reward
        self.linear_punish_reward_x = 0
        self.linear_punish_reward_y = 0

        if self.body_v.twist.linear.x < 0.2:
            self.linear_punish_reward_x = -2

        # angular punish reward
        self.angular_punish_reward = 0

        if self.body_v.twist.angular.z < -0.8:
            self.angular_punish_reward = -1
        if self.body_v.twist.angular.z > 0.8:
            self.angular_punish_reward = -1

        total_reward = distance_reward*(5/time_step)*1.2*7 \
                        + self.arrive_reward \
                        + crash_reward \
                        + laser_reward \
                        + self.linear_punish_reward_x \
                        + self.angular_punish_reward

        self.hold_able = True


        return self._cur_state(), total_reward, self.done



    # tool funciton
    def _mavrosStateCB(self, msg):
        self.mavrosState = msg

    def _scanCB(self, msg):
        self.scan = msg
    
    def _bodyVelocityCB(self, msg):
        self.body_v = msg

    def _hold(self, event):
        """
            if no cmd, hold command is sent.
        """
        if self.hold_able == False:
            return

        if (self.hold_flag == False) and ((rospy.Time.now() - self.last_cmd_time) > rospy.Duration(0.1)):
            self.hold_pose = self.pose
            self.hold_flag = True
            print("reset hold pose.")


        if self.hold_flag == True:
            # cmd_x = self.hold_pose.position.x
            if self.hold_pose.position.x > self.pose.position.x:
                cmd_x = min(self.hold_pose.position.x, self.pose.position.x+0.5)
            else:
                cmd_x = max(self.hold_pose.position.x, self.pose.position.x-0.5)
                
            # cmd_y = self.hold_pose.position.y
            if self.hold_pose.position.y > self.pose.position.y:
                cmd_y = min(self.hold_pose.position.y, self.pose.position.y+0.5)
            else:
                cmd_y = max(self.hold_pose.position.y, self.pose.position.y-0.5)

            cmd_z = self.hold_pose.position.z
            (_, _, cmd_yaw) = euler_from_quaternion([
                                                    self.hold_pose.orientation.x,
                                                    self.hold_pose.orientation.y,
                                                    self.hold_pose.orientation.z,
                                                    self.hold_pose.orientation.w
                                                    ])
            self._send_position_cmd(cmd_x, cmd_y, cmd_yaw, height=cmd_z)


    def _is_hold(self):
        return abs(self.twist.linear.x) < 10e-2 and  \
                abs(self.twist.linear.y) < 10e-2 and \
                abs(self.twist.linear.z) < 10e-2 and \
                abs(self.twist.angular.x) < 10e-2 and \
                abs(self.twist.angular.y) < 10e-2 and \
                abs(self.twist.angular.z) < 10e-2


    def _is_arrived(self, x, y, z, limit=0.1):
        return (self.pose.position.x - x)**2 < limit and \
                (self.pose.position.y - y)**2 < limit and \
                    (self.pose.position.z - z)**2 < limit

    def _vision(self, event):
        """
            get iris_0 state, and send vision.
        """
        # get stata
        req = GetModelStateRequest()
        req.model_name = self.model_name
        self.modelStateClient.wait_for_service()
        resp = self.modelStateClient.call(req)
        self.pose = resp.pose
        self.twist = resp.twist

        # send position vision
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.pose = self.pose
        self.visionPub.publish(msg)

        # send velocity vision
        msg = TwistWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.twist.twist = self.twist
        self.visionVPub.publish(msg)


    def _send_velocity_cmd(self, vx, vy, yaw_rate):
        """
            send velocity command on body end
        """
        cmd = PositionTarget()
        cmd.header.stamp = rospy.Time.now()
        cmd.coordinate_frame = PositionTarget.FRAME_BODY_NED
        cmd.type_mask = PositionTarget.IGNORE_PX | \
                        PositionTarget.IGNORE_PY | \
                        PositionTarget.IGNORE_PZ | \
                        PositionTarget.IGNORE_AFX | \
                        PositionTarget.IGNORE_AFY | \
                        PositionTarget.IGNORE_AFZ | \
                        PositionTarget.IGNORE_YAW
        cmd.velocity.x = vx
        cmd.velocity.y = vy
        cmd.velocity.z = 0.5*(self.height - self.pose.position.z)
        cmd.yaw_rate = yaw_rate

        self.ctrPub.publish(cmd)
        self.last_cmd_time = rospy.Time.now()
        self.rate.sleep()


    def _send_position_cmd(self, x, y, yaw, height = 3):
        """
            send position command on local ENU
        """
        cmd = PositionTarget()
        cmd.header.stamp = rospy.Time.now()
        cmd.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        cmd.type_mask = PositionTarget.IGNORE_VX | \
                        PositionTarget.IGNORE_VY | \
                        PositionTarget.IGNORE_VZ | \
                        PositionTarget.IGNORE_AFX | \
                        PositionTarget.IGNORE_AFY | \
                        PositionTarget.IGNORE_AFZ | \
                        PositionTarget.IGNORE_YAW_RATE
        cmd.position.x = x
        cmd.position.y = y
        cmd.position.z = height
        cmd.yaw = yaw

        self.ctrPub.publish(cmd)
        self.last_cmd_time = rospy.Time.now()
        self.rate.sleep()

    def _cur_state(self):
        """
            get current state
        """
        # ranges msg
        state = np.ones(self.state_num)
        cur_ranges = [ (i - self.scan.range_max/2)/(self.scan.range_max/2) for i in self.scan.ranges]
        for i in range(len(cur_ranges)):
            state[i] = cur_ranges[i]

        # pose msg
        state[-5] = (self.body_v.twist.linear.x - 0.5)/0.5
        state[-4] = self.body_v.twist.linear.y
        state[-3] = self.body_v.twist.angular.z/math.pi
        # relative distance and normalize
        # /10 here is to cut down magnitude of distance
        distance_uav_target =  math.sqrt((self.target_x - self.pose.position.x)**2 + (self.target_y - self.pose.position.y)**2)/10
        # distance_uav_target = (2*distance_uav_target - self.scan.range_max)/self.scan.range_max
        # relative angular difference and normalize
        angle_uav_targer = atan2(self.target_y - self.pose.position.y, self.target_x - self.pose.position.x)
        (_, _, angle_uav) = euler_from_quaternion([
                                                self.pose.orientation.x,
                                                self.pose.orientation.y,
                                                self.pose.orientation.z,
                                                self.pose.orientation.w
                                                ])
        angle_diff = angles.shortest_angular_distance(angle_uav, angle_uav_targer)/math.pi
        state[-2] = distance_uav_target
        state[-1] = angle_diff

        return state


if __name__ == '__main__':
    rospy.init_node("test")

    game = Game("iris_0")
    game.start()
    for i in range(100):    
        for j in range(50):
            o, r, done = game.step(0.1, 1, 0, 0)
            print("reward: %f, done: %d" % (r, done))
            # a, b = game.is_crashed()
            # if a == True:
            #     print("crash!")
            # for k in state:
            #     print(str(k), end=' ')
            # print("\n")
        game.reset()


    rospy.spin()