#! /usr/bin/env python
#-*- coding: UTF-8 -*- 
from math import atan2

from torch import set_flush_denormal
import rospy
from rospy.rostime import Duration

# ros include
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose, PoseStamped, TwistStamped, TwistWithCovarianceStamped
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, CommandBoolResponse
from mavros_msgs.srv import SetMode, SetModeRequest, SetModeResponse
from gazebo_msgs.srv import SetModelState, GetModelState, GetModelStateRequest, GetModelStateResponse
from gazebo_msgs.msg import ModelState 
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from multi_rotor_avoidance_rl.msg import Reward, Acc
from common.world import World
import random
import numpy as np
import math
import angles
from pykalman import KalmanFilter

class Game:

    def __init__(self, model_name, game_name):
        """
            initialize
        """


        self.model_name = model_name
        self.game_name = game_name

        self.mavrosState = State()
        self.pose = Pose()
        self.twist = Twist()
        self.scan = LaserScan()
        self.body_v = TwistStamped()

        self.crash_limit = 0.25

        self.start_flag = False

        self.target_x = 10
        self.target_y = 10

        self.state_num = 35+4
        self.action_num = 2

        self.momentum_discount = 0.4

        self.height = 2.0 # height of taking off

        self.step_count = 0

        self.rate = rospy.Rate(20)

        self.hold_flag = False # if True, send hold command
        self.hold_able = False
        self.hold_pose = Pose()
        self.last_cmd_time = rospy.Time.now()

        self.acc_x = 0
        self.acc_yaw = 0

        self.kf_x = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
        self.kf_yaw = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

        # initialize world
        if (game_name == "empty_3m"):
            self.safe_space = [[0, 0], [3, 0]]
            self.safe_radius = [1.0, 0.8]
            self.target_distance = 3
            self.wall_rate = 0.0
            self.cylinder_num = 0

        if (game_name == "train_env_3m_lite"):
            self.safe_space = [[0, 0], [3, 0]]
            self.safe_radius = [1.0, 0.8]
            self.target_distance = 3
            self.wall_rate = 0.5
            self.cylinder_num = 65

        if (game_name == "train_env_3m"):
            self.safe_space = [[0, 0], [3, 0]]
            self.safe_radius = [1.0, 0.8]
            self.target_distance = 3
            self.wall_rate = 0.8
            self.cylinder_num = 130
        
        if (game_name == "train_env_7m"):
            self.safe_space = [[0, 0], [7, 0]]
            self.safe_radius = [1.0, 0.8]
            self.target_distance = 7
            self.wall_rate = 0.8
            self.cylinder_num = 130

        if (game_name == "test_env_corridor" 
            or game_name == "test_env_cluster"):
            self.safe_space = [[0, 0]]
            self.safe_radius = [1.0]
            self.target_distance = 7
            self.wall_rate = 0
            self.cylinder_num = 0

        self.world = World(self.safe_space, self.safe_radius, wall_rate=self.wall_rate, cylinder_num=self.cylinder_num)

        # subscriber
        self.mavrosStateSub = rospy.Subscriber(self.model_name + "/mavros/state", State, self._mavrosStateCB)
        self.scanSub = rospy.Subscriber(self.model_name + "/scan_downsampled", LaserScan, self._scanCB)
        self.bodyVelocitySub = rospy.Subscriber(self.model_name + "/mavros/local_position/velocity_body", TwistStamped, self._bodyVelocityCB)
        
        # publisher
        self.ctrPub = rospy.Publisher(self.model_name + "/mavros/setpoint_raw/local", PositionTarget, queue_size=1)
        self.visionPub = rospy.Publisher(self.model_name + "/mavros/vision_pose/pose", PoseStamped, queue_size=1)
        self.visionVPub = rospy.Publisher(self.model_name + "/mavros/vision_speed/speed_twist_cov", TwistWithCovarianceStamped, queue_size=1)
        self.rewardPub = rospy.Publisher("reward", Reward, queue_size=1)
        self.accPub = rospy.Publisher("accel_body", Acc, queue_size=1)

        # service client
        self.armingClient = rospy.ServiceProxy(self.model_name + "/mavros/cmd/arming", CommandBool)
        self.setModeClient = rospy.ServiceProxy(self.model_name + "/mavros/set_mode", SetMode)
        self.modelStateClient = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

        # Timer
        self.holdTimer = rospy.Timer(rospy.Duration(0.05), self._hold)
        self.visionTimer = rospy.Timer(rospy.Duration(0.02), self._vision)

    def reset(self):
        """
            reset uav and target
        """
        self.step_count = 0

        # keep away from obstacles
        self.recovery(time=2)

        # stop in place
        while not self._is_hold():
            self._send_velocity_cmd(0, 0, 0)
            self.hold_flag = False
            self.rate.sleep()

        # holding
        self.hold_able = True
        while not self._is_hold():
            self.rate.sleep()

        # clear world
        self.world.clear()
        
        # fly home
        home_x = 0
        home_y = 0
        home_yaw = math.pi*random.uniform(-1, 1)

        while not self._is_arrived(home_x, home_y, self.height) or not self._is_hold():
            
            if home_x > self.pose.position.x:
                x_t = min(home_x, self.pose.position.x + 1)
            else:
                x_t = max(home_x, self.pose.position.x - 1)

            if home_y > self.pose.position.y:
                y_t = min(home_y, self.pose.position.y + 1)
            else:
                y_t = max(home_y, self.pose.position.y - 1)

            self._send_position_cmd(x_t, y_t, home_yaw, self.height)
            self.hold_flag = False
            self.rate.sleep()


        # holding
        self.hold_able = True
        while not self._is_hold():
            rospy.sleep(rospy.Duration(0.1))

        rospy.loginfo("initialize uav position.")

        # initialize target point
        if (self.game_name == "test_env_corridor" 
            or self.game_name == "test_env_cluster"):
            target_list = [[0, 7], [7, 0], [0, -7], [-7, 0]]
            self.target_x, self.target_y = random.choice(target_list)
        else:
            self.target_x = self.target_distance
            self.target_y = 0

        self.world.set_target(self.target_x, self.target_y)

        rospy.loginfo("initialize target position.")

        # reset world
        self.world.reset()
        rospy.loginfo("reset_world.")

        return self.cur_state()

    def start(self):
        """
            send take-off command
        """
        self.step_count = 0

        # initialize target point
        if (self.game_name == "test_env_corridor" 
            or self.game_name == "test_env_cluster"):
            target_list = [[0, 7], [7, 0], [0, -7], [-7, 0]]
            self.target_x, self.target_y = random.choice(target_list)
        else:
            self.target_x = self.target_distance
            self.target_y = 0

        self.world.set_target(self.target_x, self.target_y)

        rospy.loginfo("initialize target position.")

        # reset world
        self.world.reset()
        rospy.loginfo("reset_world.")

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

        return self.cur_state()


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
            if self.scan.ranges[i] < 3*self.crash_limit:
                self.laser_crashed_reward = min(-10, self.laser_crashed_reward)
            if self.scan.ranges[i] < 2*self.crash_limit:
                self.laser_crashed_reward = min(-25.0, self.laser_crashed_reward)
            if self.scan.ranges[i] < self.crash_limit:
                self.laser_crashed_reward = -120.0
                self.laser_crashed_flag = True
                self.crash_index = i
                break
        return self.laser_crashed_flag, self.laser_crashed_reward, self.crash_index

    def is_valid(self, state, action, time_step, limit=0.3):
        """
        determine whether action in state after time_step is valid
        return:
            - True if action is valid
            - valid reward

        dead reckoning algorithm
        """

        # restore range msg
        range_msg = np.array([ (i*self.scan.range_max + self.scan.range_max)/2 for i in state[:-6]])
        angle_msg = np.array([ self.scan.angle_min + i*self.scan.angle_increment for i in range(len(range_msg))])

        # laser position in XY coordinate system
        laser_x = range_msg*np.cos(angle_msg)
        laser_y = range_msg*np.sin(angle_msg)

        # next position in XY coordinate system
        if abs(action[1])<1e-2:
            next_x = action[0]*time_step
            next_y = 0
        else:
            radius = action[0]/action[1]
            theta = action[1]*time_step
            next_x = radius*math.sin(theta)
            next_y = radius - radius*math.cos(theta)

        # calculate distance
        distance = (laser_x - next_x)**2 + (laser_y - next_y)**2

        # check validation of action
        valid_reward = 0
        flag = True
        for i in distance:
            if i < limit**2:
                flag = False
                valid_reward = -20
                break

        return flag, valid_reward

    def step(self, time_step=0.1, vx=0.1, vy=0.1, yaw_rate=0.1):
        """
            game step
        """
        self.step_count += 1

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
        distance_reward = (last_distance - cur_distance)*(5/time_step)*1.6*7

        self.done = False

        # arrive reward
        self.arrive_reward = 0
        if cur_distance < 0.5:
            self.arrive_reward = 200
            self.done = True

        # crash reward
        crash_indicator, crash_reward, _ = self.is_crashed()
        if crash_indicator == True:
            self.done = True

        # laser reward
        state = np.array(self.scan.ranges) / float(self.scan.range_max)
        laser_reward = 0.4*(sum(state) - len(state))

        # linear punish reward (abandan)
        self.linear_punish_reward_x = 0
        # if self.body_v.twist.linear.x < 0.1:
        #     self.linear_punish_reward_x = -1

        # angular punish reward
        self.angular_punish_reward = 0
        if abs(self.body_v.twist.angular.z) > 0.4:
            self.angular_punish_reward = -1
        elif abs(self.body_v.twist.angular.z) > 0.7:
            self.angular_punish_reward = -2

        # step punish reward
        self.step_punish_reward = -self.step_count * 0.04

        # acc punish
        self.acc_x_punish_reward = -4.0*abs(self.acc_x)
        self.acc_yaw_punish_reward = -2.0*abs(self.acc_yaw)

        # right turing reward
        right_turning_reward = 0    # to break balance of turning
        if self.body_v.twist.angular.z < 0:
            right_turning_reward = 0.3*abs(self.body_v.twist.angular.z)

        total_reward = distance_reward \
                        + self.arrive_reward \
                        + crash_reward \
                        + laser_reward \
                        + self.linear_punish_reward_x \
                        + self.angular_punish_reward \
                        + self.step_punish_reward \
                        + self.acc_x_punish_reward \
                        + self.acc_yaw_punish_reward \
                        + right_turning_reward \


        msg = Reward()
        msg.header.stamp = rospy.Time.now()
        msg.distance_reward = distance_reward
        msg.arrive_reward = self.arrive_reward
        msg.crash_reward = crash_reward
        msg.laser_reward = laser_reward
        msg.linear_punish_reward = self.linear_punish_reward_x
        msg.angular_punish_reward = self.angular_punish_reward
        msg.step_punish_reward = self.step_punish_reward
        msg.acc_x_punish_reward = self.acc_x_punish_reward
        msg.acc_yaw_punish_reward = self.acc_yaw_punish_reward
        msg.right_turning_reward = right_turning_reward
        msg.total_reward = total_reward
        
        self.rewardPub.publish(msg)

        self.hold_able = True

        # out of limit
        if self.pose.position.x < -10 or self.pose.position.x > 12:
            self.done = True
        if self.pose.position.y < -10 or self.pose.position.y > 10:
            self.done = True

        return self.cur_state(), total_reward/20.0, self.done


    def recovery(self, time=1):
        """
            run recovery action (keep away from obstacles in slight velocity)
            return:
                - True if crash when recovering.
        """
        print("recovery.")
        begin_time = rospy.Time.now()
        while(rospy.Time.now() - begin_time < rospy.Duration(time)):
            crash_indicator, _, _ = self.is_crashed()
            if crash_indicator:
                return True
            index = np.argmin(self.scan.ranges)
            alpha = self.scan.angle_min + self.scan.angle_increment * index
            vx = -math.cos(alpha) * 0.1
            vy = -math.sin(alpha) * 0.1 
            self._send_velocity_cmd(vx, vy, 0)
            self.hold_flag = False
            self.rate.sleep()

        return False

    # tool funciton
    def _mavrosStateCB(self, msg):
        self.mavrosState = msg

    def _scanCB(self, msg):
        self.scan = msg
    
    def _bodyVelocityCB(self, msg):
        # save data
        self.acc_x = self.kf_x.filter(msg.twist.linear.x- self.body_v.twist.linear.x)[0][0][0]*30
        self.acc_yaw = self.kf_yaw.filter(msg.twist.angular.z - self.body_v.twist.angular.z)[0][0][0]*30
        self.body_v = msg

        # pub acc
        msg = Acc()
        msg.header.stamp = rospy.Time.now()
        msg.acc_x = self.acc_x
        msg.acc_yaw = self.acc_yaw
        self.accPub.publish(msg)

    def _hold(self, event):
        """
            if no cmd, hold command is sent.
        """
        if self.hold_able == False:
            return

        if (self.hold_flag == False) and ((rospy.Time.now() - self.last_cmd_time) > rospy.Duration(0.2)):
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
            get iris state, and send vision.
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

    def cur_state(self):
        """
            35 laser
            1  vx
            1  yaw_rate
            1  distance
            1  angle_diff
        """
        # ranges msg
        state = [ (i - self.scan.range_max/2)/(self.scan.range_max/2) for i in self.scan.ranges]

        # pose msg
        state.append(self.body_v.twist.linear.x/0.5)
        state.append(self.body_v.twist.angular.z)
        state.append(self.acc_x)
        state.append(self.acc_yaw)

        # relative distance and normalize
        distance_uav_target =  math.sqrt((self.target_x - self.pose.position.x)**2 + (self.target_y - self.pose.position.y)**2)/10
        angle_uav_targer = atan2(self.target_y - self.pose.position.y, self.target_x - self.pose.position.x)
        (_, _, angle_uav) = euler_from_quaternion([
                                                self.pose.orientation.x,
                                                self.pose.orientation.y,
                                                self.pose.orientation.z,
                                                self.pose.orientation.w
                                                ])
        angle_diff = angles.shortest_angular_distance(angle_uav, angle_uav_targer)/math.pi
        state.append(distance_uav_target)
        state.append(angle_diff)

        return state


if __name__ == '__main__':
    rospy.init_node("test")

    game = Game("iris", "empty_3m")
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