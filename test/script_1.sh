#! /bin/bash 
source /opt/ros/melodic/setup.bash
source ~/software_static/PX4_Firmware/Tools/setup_gazebo.bash ~/software_static/PX4_Firmware/ ~/software_static/PX4_Firmware/build/px4_sitl_default
source ~/software_var/catkin_ws/devel/setup.bash
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/software_static/PX4_Firmware
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/software_static/PX4_Firmware/Tools/sitl_gazebo
source ~/anaconda3/bin/activate my_env
roslaunch multi_rotor_avoidance_rl setup_env.launch
