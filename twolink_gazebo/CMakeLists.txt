cmake_minimum_required(VERSION 3.8)
project(twolink_gazebo)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  urdf
  config
  meshes
  worlds
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/dummy_data_publisher.py
  scripts/joint_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
