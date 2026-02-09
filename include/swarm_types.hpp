#include <ros/ros.h>
#include <iostream>
#include <stdlib.h>
#include <thread>
#include <mutex>
#include <queue>
#include <Eigen/Eigen>
#include <execution>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<PointType> PointCloud;

namespace DataManage{
  enum RoboType{
    GROUND = 0,
    AIR = 1
  };

  class DataManager
  {
  private:
    int self_id_;
    nav_msgs::Path path_msg; 
    std::mutex odom_mtx, laser_mtx;

  public:
    typedef std::shared_ptr<DataManager> Ptr;

    int robo_id;
    int frame_idx_ = -1;
    DataManage::RoboType robo_type_;
    ros::Publisher cloud_pub_, odom_pub_, path_pub_;
    ros::Subscriber laser_sub_, odom_sub_;

    std::vector<Eigen::Isometry3d> odom_poses_vec;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_vec;
    std::vector<std::string> cloud_path_vec;
    std::vector<double> odom_times_vec;
    std::vector<Eigen::Quaterniond> imu_vec;
    std::vector<double> imu_times;
    double newest_timestamp = 0;
    bool have_msg = false;
    //online queue
    std::queue<sensor_msgs::PointCloud2::ConstPtr> laser_buffer;
    std::queue<nav_msgs::Odometry::ConstPtr> odom_buffer;
    
  public:

    DataManager() = default;  
    DataManager(int id) : self_id_(id) {} 

    void init(const int self_id , int type, std::string frame = "world")
    {
      self_id_ = self_id;
      robo_type_  = static_cast<RoboType>(type);
      path_msg.header.frame_id = frame; 
    }

    void laserCloudHandler(const sensor_msgs::PointCloud2::ConstPtr &msg){
      std::lock_guard<std::mutex> lock(laser_mtx);

      laser_buffer.push(msg);
    }

    void odomHandler(const nav_msgs::Odometry::ConstPtr &msg) {
      std::lock_guard<std::mutex> lock(odom_mtx);
      if(!have_msg) have_msg = true;
      odom_buffer.push(msg);
    }

    bool syncPackages(PointCloud::Ptr &cloud, Eigen::Isometry3d &pose, ros::Time &time) {
      if (laser_buffer.empty() || odom_buffer.empty())
        return false;

      auto laser_msg = laser_buffer.front();
      double laser_timestamp = laser_msg->header.stamp.toSec();

      auto odom_msg = odom_buffer.front();
      double odom_timestamp = odom_msg->header.stamp.toSec();

      // check if timestamps are matched
      if (abs(odom_timestamp - laser_timestamp) < 1e-3) {
        pcl::fromROSMsg(*laser_msg, *cloud);

        Eigen::Quaterniond r(
            odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x,
            odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z);
        Eigen::Vector3d t(odom_msg->pose.pose.position.x,
                          odom_msg->pose.pose.position.y,
                          odom_msg->pose.pose.position.z);

        pose = Eigen::Isometry3d::Identity();
        pose.translate(t);
        pose.rotate(r);

        std::lock_guard<std::mutex> l_lock(laser_mtx);
        std::lock_guard<std::mutex> o_lock(odom_mtx);

        time = odom_msg->header.stamp;
        laser_buffer.pop();
        odom_buffer.pop();

      } else if (odom_timestamp < laser_timestamp) {
        ROS_WARN("Current odometry is earlier than laser scan, discard one "
                "odometry data.");
        std::lock_guard<std::mutex> o_lock(odom_mtx);
        odom_buffer.pop();
        return false;
      } else {
        ROS_WARN(
            "Current laser scan is earlier than odometry, discard one laser scan.");
        std::lock_guard<std::mutex> l_lock(laser_mtx);
        laser_buffer.pop();
        return false;
      }

      frame_idx_++;
      return true;
    }

    void updatePathMsg(Eigen::Isometry3d pose){
      path_msg.header.stamp = ros::Time::now();
      geometry_msgs::PoseStamped pose_stamped;
      pose_stamped.header.frame_id = path_msg.header.frame_id ;
      pose_stamped.header.stamp = ros::Time::now();
      
      pose_stamped.pose.position.x = pose.translation().x();
      pose_stamped.pose.position.y = pose.translation().y();
      pose_stamped.pose.position.z = pose.translation().z();

      Eigen::Quaterniond q(pose.rotation());
      pose_stamped.pose.orientation.x = q.x();
      pose_stamped.pose.orientation.y = q.y();
      pose_stamped.pose.orientation.z = q.z();
      pose_stamped.pose.orientation.w = q.w();

      path_msg.poses.push_back(pose_stamped);
      path_pub_.publish(path_msg);
    }

    void updatePathMsg(std::vector<Eigen::Isometry3d>::iterator start, 
                        std::vector<Eigen::Isometry3d>::iterator end){
      path_msg.poses.clear();
      std::for_each(start, end, [&](Eigen::Isometry3d pose) {
        path_msg.header.stamp = ros::Time::now();
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.frame_id = path_msg.header.frame_id ;
        pose_stamped.header.stamp = ros::Time::now();
        
        pose_stamped.pose.position.x = pose.translation().x();
        pose_stamped.pose.position.y = pose.translation().y();
        pose_stamped.pose.position.z = pose.translation().z();

        Eigen::Quaterniond q(pose.rotation());
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        path_msg.poses.push_back(pose_stamped);
      });
      path_pub_.publish(path_msg);
    }
  };
}