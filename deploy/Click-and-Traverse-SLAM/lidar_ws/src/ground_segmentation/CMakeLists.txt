cmake_minimum_required(VERSION 3.5)
project(ground_segmentation)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)

# Add executable
add_executable(ground_segmentation_node src/ground_segmentation_node.cpp)

# Link PCL and other dependencies
ament_target_dependencies(ground_segmentation_node
  rclcpp
  sensor_msgs
  PCL
  pcl_conversions
)

target_link_libraries(ground_segmentation_node ${PCL_LIBRARIES})

# Install the target
install(TARGETS
  ground_segmentation_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install package.xml
install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

# Install CMakeLists.txt
install(FILES CMakeLists.txt
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

