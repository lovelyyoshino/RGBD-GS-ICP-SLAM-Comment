cmake_minimum_required(VERSION 3.0.2)
project(gs_icp_slam)

# 查找所需的catkin包
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  message_filters
  cv_bridge
)

catkin_package()

# 安装脚本
catkin_install_python(PROGRAMS
  ros_gs_icp.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
