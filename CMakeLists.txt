cmake_minimum_required(VERSION 3.0.2)
project(ehpr)

set(CMAKE_BUILD_TYPE "Release")

# set(CMAKE_BUILD_TYPE "Debug")
# Enable debug flags (use if you want to debug in gdb)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized")

set(SRRG_HBST_HAS_EIGEN true)
add_definitions(-DSRRG_HBST_HAS_EIGEN)
set(SRRG_HBST_HAS_OPENCV true)
add_definitions(-DSRRG_HBST_HAS_OPENCV)
add_definitions(-DSRRG_MERGE_DESCRIPTORS)

add_definitions(-DPJSRCDIR=\"${PROJECT_SOURCE_DIR}\")
add_definitions(-DSAVE_MID_FILE=1)   # bool: whether to save intermediate files (unnecessary to the program)

find_package(catkin REQUIRED COMPONENTS
        geometry_msgs
        nav_msgs
        sensor_msgs
        roscpp
        rospy
        rosbag
        std_msgs
        tf
        eigen_conversions
        )

find_package(PkgConfig REQUIRED)
pkg_check_modules(glog REQUIRED libglog)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(TBB REQUIRED)
find_package(GTSAM REQUIRED QUIET)

include_directories(
        include
        thirdparty
        ${glog_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TBB_INCLUDE_DIRS}
)

catkin_package(
        CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy rosbag std_msgs 
        DEPENDS EIGEN3 PCL
        INCLUDE_DIRS include
)

#ehpr library
add_library(${PROJECT_NAME}lib
        src/ehpr/desc_mng.cpp
        src/ehpr/desc_db.cpp
        src/ehpr/AlignRansac2D.cpp
        )
target_link_libraries(${PROJECT_NAME}lib
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${glog_LIBRARIES}
        ${OpenCV_LIBS}
        ${TBB_LIBRARIES}
        )

add_executable(${PROJECT_NAME}_multisession_video
        test/multisession_video.cpp
        )
target_link_libraries(${PROJECT_NAME}_multisession_video
        ${PROJECT_NAME}lib
        ${catkin_LIBRARIES}
        gtsam
        )
