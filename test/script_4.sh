#! /bin/bash 
source /opt/ros/melodic/setup.bash
source ~/software_var/catkin_ws/devel/setup.bash
source ~/anaconda3/bin/activate my_env
rosrun multi_rotor_avoidance_rl var_collect.py
