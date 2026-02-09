#include <memory>
#include <utility>
#include <thread>
#include <chrono>
#include <filesystem>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include "ehpr/desc_db.h"
#include "tools/config_handler.h"
#include "data_tools.hpp"
#include "swarm_types.hpp"
#include "VoxelMap.hpp"

using namespace std;

const std::string PROJ_DIR = std::string(PJSRCDIR);
SequentialTimeProfiler stp;

//for gtsam index
int CloudInd = 0;
int keyCloudInd = 0;
std::vector<int> key_idx_vec; //key frame in cloudind
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_vec_;

//for data
std::unique_ptr<Datatools> ptr_datatools;
std::vector<DataManage::DataManager::Ptr> datamng_;
int ground_robonum = 0;
int air_robonum = 0;

//for voxel map
VoxelMapConfig map_config;
std::shared_ptr<VoxelMap> local_map_;
pcl::PointCloud<pcl::PointXYZ>::Ptr key_cloud;
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> key_cloud_vec_;
Eigen::Vector3d last_pos;
std::vector<Eigen::Isometry3d> key_odom_poses;
std::vector<Eigen::Quaterniond>key_grav_ests;
int imu_en_ = 0; 
double travel_dis = 0.0;
bool record_keypose = false;

//imu correct 
Eigen::Isometry3d correct_est;
bool have_correct = false;

//for register 
double inlier_thre_, fit_thre_;
double inlier_thre_gg, inlier_thre_he;
double fit_thre_gg, fit_thre_he;
DescriptorManagerConfig mg_config;
std::unique_ptr<DescriptorDB> ptr_desc_db;
//fot test
int loop_en_= 0;

//for display
std::vector<Eigen::Vector3d> colors(5);
std::vector<std::pair<int, int>> loop_pairs;

void visualizekeyframe(    
  const int robo_id,
  const int local_id,
  const Eigen::Isometry3d& odom_now,
  const Eigen::Isometry3d& odom_last,
  const Eigen::Vector3d& color,
  const ros::Publisher& graph_pub){
  visualization_msgs::Marker point;
  point.header.frame_id = "camera_init";
  point.header.stamp = ros::Time::now();
  point.ns = "keypoints"+std::to_string(robo_id);
  point.id = local_id;
  point.type = visualization_msgs::Marker::SPHERE;
  point.action = visualization_msgs::Marker::ADD;
  point.scale.x = 2.5;
  point.scale.y = 2.5;
  point.scale.z = 2.5;
  point.color.r = color(0);
  point.color.g = color(1);
  point.color.b = color(2);
  point.color.a = 1.0;

  // 获取需要发布的点
  geometry_msgs::Point pt;
  pt.x = odom_now.translation().x();
  pt.y = odom_now.translation().y();
  pt.z = odom_now.translation().z();
  point.pose.position = pt;  // 设置点的位置
  // 发布点
  graph_pub.publish(point);
}

//visual edge of loop
void visualizeloop(    
  // const std::vector<Eigen::Isometry3d>& odom_vec,
  // const std::pair <int, int>loop_pair,
  const Eigen::Isometry3d& odom_now,
  const Eigen::Isometry3d& odom_last,
  const int loop_id,
  const ros::Publisher& graph_pub,
  bool clear_com=false){
    visualization_msgs::Marker loop_edges;
    loop_edges.header.frame_id = "camera_init";
    loop_edges.header.stamp = ros::Time::now();
    loop_edges.ns = "loopedges";
    loop_edges.id = loop_id;
    if(!clear_com){
      loop_edges.type = visualization_msgs::Marker::LINE_LIST;
      loop_edges.action = visualization_msgs::Marker::ADD;
      loop_edges.scale.x = 0.6; // 线的宽度

      loop_edges.color.r = 1.0;
      loop_edges.color.g = 1.0;
      loop_edges.color.b = 0.0;
      loop_edges.color.a = 0.7;

      geometry_msgs::Point p1, p2;
      Eigen::Isometry3d pose1 = odom_now;
      Eigen::Isometry3d pose2 = odom_last;

      p1.x = pose1.translation().x();
      p1.y = pose1.translation().y();
      p1.z = pose1.translation().z();

      p2.x = pose2.translation().x();
      p2.y = pose2.translation().y();
      p2.z = pose2.translation().z();

      loop_edges.points.push_back(p1);
      loop_edges.points.push_back(p2);
    }
    else{
      loop_edges.action = visualization_msgs::Marker::DELETE;
    }
    graph_pub.publish(loop_edges);
}

void visualizeGraph(
    const std::vector<int>& key_vec,
    const std::vector<Eigen::Isometry3d>& odom_vec,
    const std::vector<std::pair<int, int>>& loop_vec,
    const std::vector<int>& keyaccumsize_vec,
    const ros::Publisher& loop_pub,
    const ros::Publisher& key_pub){   
    // 创建并设置关键点的Marker
    visualization_msgs::Marker points, lines;
    for(int session_id=0; session_id<keyaccumsize_vec.size(); session_id++){
      int begin, end;
      if(session_id == 0) begin = 0;
      else begin = keyaccumsize_vec[session_id-1]+1;
      end = keyaccumsize_vec[session_id];
      for(int accum_id = begin; accum_id <= end; accum_id++){
        Eigen::Isometry3d prev_pose;
        if(accum_id-begin > 0){
          prev_pose = odom_vec[key_vec[accum_id-1]];
        }
        visualizekeyframe(session_id, accum_id-begin, odom_vec[key_vec[accum_id]], prev_pose, colors[session_id], key_pub);
      }
    }

    int loop_ind = 0;
    for (const auto& loop_pair : loop_vec) {
      int global_idx1 = key_vec[loop_pair.first];
      int global_idx2 = key_vec[loop_pair.second];
      Eigen::Isometry3d pose1 = odom_vec[global_idx1];
      Eigen::Isometry3d pose2 = odom_vec[global_idx2];
      visualizeloop(pose1, pose2, loop_ind, loop_pub);
      loop_ind++;
    }
}

void update_poses(const gtsam::Values &estimates,
                  std::vector<Eigen::Isometry3d> &poses) {
  CHECK_EQ(estimates.size(), poses.size());
  poses.clear();
  for (int i = 0; i < estimates.size(); ++i) {
    auto est = estimates.at<gtsam::Pose3>(i);
    Eigen::Isometry3d est_affine3d(est.matrix());
    est_affine3d = correct_est * est_affine3d;
    poses.push_back(est_affine3d);
  }
}

void publishOdometry(const Eigen::Isometry3d& pose, const double time, ros::Publisher& pub) {
    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = ros::Time().fromSec(time); 
    odom_msg.header.frame_id = "camera_init"; 
    odom_msg.child_frame_id = "base_link";  

    Eigen::Quaterniond quat(pose.rotation());
    Eigen::Vector3d trans = pose.translation();  
    odom_msg.pose.pose.position.x = trans.x();
    odom_msg.pose.pose.position.y = trans.y();
    odom_msg.pose.pose.position.z = trans.z();
    odom_msg.pose.pose.orientation.x = quat.x();
    odom_msg.pose.pose.orientation.y = quat.y();
    odom_msg.pose.pose.orientation.z = quat.z();
    odom_msg.pose.pose.orientation.w = quat.w();

    pub.publish(odom_msg);
}

double fineRegistration(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &target,
    const Eigen::Matrix4f &initial_pose,
    Eigen::Matrix4f &final_pose,
    const std::string &output_filename = "")
{

  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());

  // 点云预处理
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  float voxel = 0.8f; // 0.8
  sor.setInputCloud(source);
  sor.setLeafSize(voxel, voxel, voxel);
  sor.filter(*source_cloud);
  std::cout << "source_cloud " << source_cloud->points.size() << std::endl;

  sor.setInputCloud(target);
  sor.setLeafSize(voxel, voxel, voxel);
  sor.filter(*target_cloud);
  std::cout << "target_cloud " << target_cloud->points.size() << std::endl;

  // 配准结果存储
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>());

  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
  // pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
  gicp.setMaximumIterations(10);
  gicp.setMaxCorrespondenceDistance(5); // 10
  gicp.setInputSource(source_cloud);
  gicp.setInputTarget(target_cloud);
  gicp.align(*aligned_cloud, initial_pose);

  if (gicp.hasConverged())
  {
    final_pose = gicp.getFinalTransformation();
    std::cout << "--- T_target_source ---\n"
              << final_pose << std::endl;
    std::cout << "ICP has converged, score: " << gicp.getFitnessScore(20) << std::endl;
    if (!output_filename.empty())
    {
      // 创建 RGB 点云以保存颜色信息
    }
    return gicp.getFitnessScore(20);
  }
  else
  {
    std::cout << "ICP did not converge." << gicp.getFitnessScore(20) << std::endl;
    return -1;
    ;
  }
}

bool loadConfig(const std::string &config_fpath, std::string &sav_path) {
  auto yl = yamlLoader(config_fpath);
  key_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
  ptr_datatools = std::make_unique<Datatools>();
  Eigen::Vector3d groud_est(-0.0149687, -0.012568, 0.999809);
  Eigen::Vector3d target_normal(0.0, 0.0, 1.0);
  Eigen::Vector3d rotation_axis = groud_est.cross(target_normal);
  double rotation_angle = acos(groud_est.dot(target_normal));
  Eigen::AngleAxisd rotation_vector(rotation_angle, rotation_axis.normalized());
  Eigen::Matrix3d rotation_matrix = rotation_vector.toRotationMatrix();
  Eigen::Matrix4d transform_matrix = Eigen::Matrix4d::Identity();
  transform_matrix.block<3, 3>(0, 0) = rotation_matrix; 
  correct_est =  Eigen::Isometry3d(transform_matrix);

  colors[0](0)= 0.0; colors[0](1) = 1.0; colors[0](2)= 0.0;  
  colors[1](0)= 1.0; colors[1](1) = 0.0; colors[1](2)= 0.0;  
  colors[2](0)= 0.0; colors[2](1) = 0.0; colors[2](2)= 0.8; 
  colors[3](0)= 1.0; colors[3](1) = 1.0; colors[3](2)= 0.0; 
  colors[4](0)= 0.0; colors[4](1) = 1.0; colors[4](2)= 1.0;

  printf("Loading parameters...\n");

  yl.loadOneConfig({"SwarmParam", "ground_robonum"}, ground_robonum);
  yl.loadOneConfig({"SwarmParam", "air_robonum"}, air_robonum);
  yl.loadOneConfig({"SwarmParam", "inlier_thre_gg"}, inlier_thre_gg);
  yl.loadOneConfig({"SwarmParam", "inlier_thre_he"}, inlier_thre_he);
  yl.loadOneConfig({"SwarmParam", "fit_thre_gg"}, fit_thre_gg);
  yl.loadOneConfig({"SwarmParam", "fit_thre_he"}, fit_thre_he);

  datamng_.resize(ground_robonum+air_robonum); 
  for (int i = 0; i < datamng_.size(); ++i)
    datamng_[i] = std::make_shared<DataManage::DataManager>();

  yl.loadOneConfig({"EnableParam", "loop_en_"}, loop_en_);
  yl.loadOneConfig({"EnableParam", "imu_en_"}, imu_en_);

  for(int i=0; i<ground_robonum; i++){
    datamng_[i]->init(i, 0, "camera_init");
    std::vector<Eigen::Isometry3d> tmp_poses;
    std::vector<double> tmp_times;
    std::string root_path;// odom_path_, lidar_path_, imu_path_;
    yl.loadOneConfig({"Path", "ground_root"+std::to_string(i)}, root_path);  
    Datatools::loadPoses(std::string(root_path+"register_pose.txt"), datamng_[i]->odom_poses_vec, datamng_[i]->odom_times_vec);
    if(imu_en_){
      Datatools::loadIMUs(std::string(root_path+"raw_imu_data.txt"),  datamng_[i]->imu_vec, datamng_[i]->imu_times);
    }
    Datatools::loadCloudsPath(std::string(root_path+"register_lidar.txt"), datamng_[i]->cloud_path_vec, datamng_[i]->odom_poses_vec, root_path+"pcd/");
  }

  for(int i=0; i<air_robonum; i++){
    int mng_idx = i+ground_robonum;
    datamng_[mng_idx]->init(mng_idx, 1, "camera_init");
    std::vector<Eigen::Isometry3d> tmp_poses;
    std::vector<double> tmp_times;
    std::string root_path;// odom_path_, lidar_path_, imu_path_;
    yl.loadOneConfig({"Path", "air_root"+std::to_string(i)}, root_path);  
    Datatools::loadPoses(std::string(root_path+"register_pose.txt"), datamng_[mng_idx]->odom_poses_vec, datamng_[mng_idx]->odom_times_vec);
    if(imu_en_){
      Datatools::loadIMUs(std::string(root_path+"raw_imu_data.txt"),  datamng_[mng_idx]->imu_vec, datamng_[mng_idx]->imu_times);
    }
    Datatools::loadCloudsPath(std::string(root_path+"register_lidar.txt"), datamng_[mng_idx]->cloud_path_vec, datamng_[mng_idx]->odom_poses_vec, root_path+"pcd/");
  }

  yl.loadOneConfig({"VoxelMapBConfig", "voxel_size"}, map_config.voxel_size);
  yl.loadOneConfig({"VoxelMapBConfig", "max_distance"}, map_config.max_distance);
  yl.loadOneConfig({"VoxelMapBConfig", "remove_distance"}, map_config.remove_distance);
  yl.loadOneConfig({"VoxelMapBConfig", "max_points_per_voxel"}, map_config.max_points_per_voxel);
  local_map_.reset(new VoxelMap(map_config));

  //!for ehpr
  DescriptorDBConfig db_config;
  yl.loadOneConfig({"DescriptorDBConfig", "bucket_size_"}, db_config.bucket_size_);
  yl.loadOneConfig({"DescriptorDBConfig", "distri_mean_"}, db_config.distri_mean_);
  yl.loadOneConfig({"DescriptorDBConfig", "distri_var_"}, db_config.distri_var_);
  yl.loadOneConfig({"DescriptorDBConfig", "dist_thres_"}, db_config.dist_thres_);

  yl.loadOneConfig({"DescriptorManagerConfig", "reso_row_"}, db_config.reso_row_);
  yl.loadOneConfig({"DescriptorManagerConfig", "reso_col_"}, db_config.reso_col_);
  yl.loadOneConfig({"DescriptorManagerConfig", "n_row_"}, db_config.n_row_);
  yl.loadOneConfig({"DescriptorManagerConfig", "n_col_"}, db_config.n_col_);
  yl.loadOneConfig({"DescriptorDBConfig", "skip_num_"}, db_config.skip_num_);

  ptr_desc_db = std::make_unique<DescriptorDB>(db_config);

  yl.loadOneConfig({"DescriptorManagerConfig", "reso_row_"}, mg_config.reso_row_);
  yl.loadOneConfig({"DescriptorManagerConfig", "reso_col_"}, mg_config.reso_col_);
  yl.loadOneConfig({"DescriptorManagerConfig", "n_row_"}, mg_config.n_row_);
  yl.loadOneConfig({"DescriptorManagerConfig", "n_col_"}, mg_config.n_col_);
  yl.loadOneConfig({"DescriptorManagerConfig", "local_density_rows_"}, mg_config.local_density_rows_);
  yl.loadOneConfig({"DescriptorManagerConfig", "local_density_cols_"}, mg_config.local_density_cols_);
  yl.loadOneConfig({"DescriptorManagerConfig", "max_dens_"}, mg_config.max_dens_);

  yl.loadOneConfig({"Path", "fpath_outcome_sav"}, sav_path);
  yl.close();

  return true;
}
bool generateSubmap(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const Eigen::Isometry3d &pose,
                     const Eigen::Quaterniond &rotation = Eigen::Quaterniond::Identity()){
  local_map_->addPoints(cloud);
  Eigen::Vector3d cur_pos = pose.translation() ;
  travel_dis += (cur_pos - last_pos).norm();
  last_pos = cur_pos;

  if(!record_keypose && travel_dis >= map_config.max_distance*0.01){
    Eigen::Quaterniond self_quaternion = rotation;
    Eigen::Vector3d eulerAngle = self_quaternion.matrix().eulerAngles(2, 1, 0);
    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(eulerAngle(2), Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(eulerAngle(1), Eigen::Vector3d::UnitY()));
    Eigen::Quaterniond rotation_yx = pitchAngle * rollAngle;
    key_odom_poses.emplace_back(pose);
    key_grav_ests.emplace_back(rotation_yx);
    // gravity_est.rotate(rotation_yx);
    key_idx_vec.emplace_back(CloudInd);
    record_keypose = true;
  }

  if(travel_dis > map_config.max_distance && record_keypose){
    local_map_->getAllPoints(key_cloud);
    Eigen::Affine3d key_origin = key_odom_poses.back();

    pcl::transformPointCloud(*key_cloud, *key_cloud, key_origin.inverse());
    #if SAVE_MID_FILE
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_save;
      cloud_save = key_cloud;
      std::string f_name = PROJ_DIR + "/results/keycloud/keycloud_" +
                            std::to_string(keyCloudInd) + ".pcd";
      pcl::io::savePCDFileASCII(f_name, *cloud_save);
    #endif
    key_cloud_vec_.push_back(key_cloud->makeShared());
    Eigen::Affine3f Tg = Eigen::Affine3f::Identity();
    Tg.linear() = key_grav_ests.back().cast<float>().toRotationMatrix();
    pcl::transformPointCloud(*key_cloud, *key_cloud, Tg);

    //prepare for next
    local_map_->removePointsFarFromLocation(cur_pos);
    record_keypose = false;
    travel_dis = 0.0;
    return true;
  }
  return false;
}

Eigen::Isometry3d adjustPoseZAxis(const Eigen::Isometry3d& pose_input) {
    Eigen::Isometry3d pose = pose_input; // 拷贝输入的 pose
    Eigen::Vector3d translation = pose.translation();
    translation.z() += 30;
    pose.translation() = translation;
    return pose;
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  ros::init(argc, argv, "demo_multisession");
  ros::NodeHandle nh;
  //-----load pose and data----//
  std::string cand_score_config = PROJ_DIR + "/config/multisession_video.yaml";
  std::cout<<cand_score_config<<std::endl;
  std::string fpath_outcome_sav;

  ros::Publisher pubOriginCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_origin", 10000);
  ros::Publisher pubCorrectCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_correct", 9999999999);
  ros::Publisher pubOdomOrigin = nh.advertise<nav_msgs::Odometry>("/odom_origin", 10);
  ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/opt_path", 10);
  ros::Publisher pubEdge = nh.advertise<visualization_msgs::Marker>("/edges", 100000);
  ros::Publisher pubKey = nh.advertise<visualization_msgs::Marker>("/keypoints", 100000);
  
  loadConfig(cand_score_config, fpath_outcome_sav);
  stp = SequentialTimeProfiler(fpath_outcome_sav);

  for (int i = 0; i < ground_robonum+air_robonum; i++){
    std::string pub_odom_name = std::string("/odom_topic") + std::to_string(i);
    std::string pub_cloud_name = std::string("/cloud_topic") + std::to_string(i);
    std::string pub_path_name = std::string("/path_topic") + std::to_string(i);
    
    datamng_[i]->odom_pub_ = nh.advertise<nav_msgs::Odometry>(pub_odom_name, 100);
    datamng_[i]->cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>(pub_cloud_name, 100);
    datamng_[i]->path_pub_ = nh.advertise<nav_msgs::Path>(pub_path_name, 100);
  }

  printf("\nHold for 3 seconds...\n");
  std::this_thread::sleep_for(std::chrono::duration<double>(3.0)); 

  //---gtsam---/
  gtsam::Values initial;
  gtsam::NonlinearFactorGraph graph;
  gtsam::noiseModel::Diagonal::shared_ptr odometryNoise =
      gtsam::noiseModel::Diagonal::Variances(
          // (gtsam::Vector(6) << 1e-9, 1e-9, 1e-9, 1e-6, 1e-6, 1e-6).finished());
          (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
          // (gtsam::Vector(6) << 1e-8, 1e-8, 1e-8, 1e-5, 1e-5, 1e-5).finished());

  double loopNoiseScore = 1e-4;
  gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)

  gtsam::ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.01;
  parameters.relinearizeSkip = 1;
  gtsam::ISAM2 isam(parameters);
  gtsam::Values curr_estimate;
  //---gtsam---//

  std::vector<Eigen::Isometry3d> global_odom_poses;//global odom
  std::vector<double> global_time;//global time
  std::vector<int> key_accumsize_vec;
  std::vector<int> ind_accumsize_vec;
  bool has_loop_flag = false;
  int has_inter_loop;
  pcl::PointCloud<pcl::PointXYZ>::Ptr full_map(new pcl::PointCloud<pcl::PointXYZ>);

  for(int sess_id = 0; sess_id<ground_robonum+air_robonum; sess_id++){
    Eigen::Isometry3d world_relative_pose = Eigen::Isometry3d::Identity();
    has_inter_loop = 0;
    last_pos = datamng_[sess_id]->odom_poses_vec[0].translation();
    printf("\n next session with id %d...\n", sess_id);
    for(int index = 0; index < datamng_[sess_id]->odom_poses_vec.size(); index++){
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_body(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::io::loadPCDFile<pcl::PointXYZ>(datamng_[sess_id]->cloud_path_vec[index], *cloud);
      Eigen::Isometry3d pose = datamng_[sess_id]->odom_poses_vec[index];
      Eigen::Isometry3d visual_pose = pose;
      if(sess_id > 0) visual_pose = adjustPoseZAxis(pose);
      //cloud_vec.emplace_back
      // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = datamng_[sess_id].cloud_vec[index];
      pcl::transformPointCloud(*cloud, *cloud_body, Eigen::Affine3d(pose.inverse()));
      global_odom_poses.emplace_back(pose);
      global_time.emplace_back(datamng_[sess_id]->odom_times_vec[index]);
      cloud_vec_.emplace_back(cloud_body);

      datamng_[sess_id]->updatePathMsg(visual_pose);
      publishOdometry(pose, datamng_[sess_id]->odom_times_vec[index], pubOdomOrigin);

      sensor_msgs::PointCloud2 pub_cloud;
      pcl::PointCloud<pcl::PointXYZ> cloud_visual;
      pcl::transformPointCloud(*cloud_body, cloud_visual, Eigen::Affine3d(visual_pose));
      pcl::toROSMsg(cloud_visual, pub_cloud);
      pub_cloud.header.frame_id = "camera_init";
      pubOriginCloud.publish(pub_cloud);

      if(!loop_en_) continue;

      initial.insert(
        CloudInd, 
        gtsam::Pose3((pose).matrix())); //initial correct with loop
      if (!CloudInd) {
        graph.add(gtsam::PriorFactor<gtsam::Pose3>(
            0, 
            gtsam::Pose3(pose.matrix()), 
            odometryNoise));
      } else {
        if(index != 0){
          auto prev_pose = gtsam::Pose3(datamng_[sess_id]->odom_poses_vec[index - 1].matrix());
          auto curr_pose = gtsam::Pose3(pose.matrix());
          graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
            CloudInd - 1, 
            CloudInd, 
            prev_pose.between(curr_pose),
            odometryNoise));
        }
        else{
        }
      }
      
      double cur_time = datamng_[sess_id]->odom_times_vec[index];
      Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
      if(imu_en_){
        int imu_idx = lookupNN<double>(cur_time, datamng_[sess_id]->imu_times, 0.03);
        rotation = datamng_[sess_id]->imu_vec[imu_idx];
      }
      // if(!have_correct){
      //   Eigen::Quaterniond self_quaternion = rotation;
      //   Eigen::Vector3d eulerAngle = self_quaternion.matrix().eulerAngles(2, 1, 0);
      //   Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(eulerAngle(2), Eigen::Vector3d::UnitX()));
      //   Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(eulerAngle(1), Eigen::Vector3d::UnitY()));
      //   Eigen::Quaterniond rotation_yx = pitchAngle * rollAngle;
      //   correct_est = correct_est*rotation_yx;
      //   have_correct =  true;
      // }
      if(generateSubmap(cloud, pose, rotation)){
        std::cout << "\nkeyCloudInd========" << keyCloudInd << std::endl;
        Eigen::Isometry3d prev_pose; 
        int key_local_id;
        int accum_id = 0;
        if(sess_id > 0){
          accum_id = key_accumsize_vec[sess_id-1]+1;
        }
        key_local_id = keyCloudInd - accum_id;
        if(key_local_id > 0){
          prev_pose = key_odom_poses[key_odom_poses.size() - 2];
        }
        Eigen::Isometry3d visual_pose = pose;
        if(sess_id > 0)
          visualizekeyframe(sess_id, key_local_id, adjustPoseZAxis(key_odom_poses.back()), adjustPoseZAxis(prev_pose), colors[sess_id], pubKey);
        else visualizekeyframe(sess_id, key_local_id, key_odom_poses.back(), prev_pose, colors[sess_id], pubKey);

        //----!step1. Descriptor Extraction----//
        std::shared_ptr<DescriptorManager> dmng_ptr(new DescriptorManager(mg_config, keyCloudInd));
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr out_ptr = key_cloud;
        stp.lap();
        stp.start();
        dmng_ptr->makeDensityBEV<pcl::PointXYZ>(out_ptr, std::to_string(keyCloudInd));
        stp.record("makeBEVandHeightBin");

        stp.start();
        dmng_ptr->makeFeature();
        ptr_desc_db->getClassMatchables(dmng_ptr);
        stp.record("makeDescriptor");
        dmng_ptr->clearImage(); 

        //-------!step2. Searching Loop----//
        std::vector<double> cand_corr;
        std::vector<int> cand_indx;
        Eigen::Matrix4d initial_pose;
        Eigen::Matrix4f final_pose;
        cand_corr.clear();
        cand_indx.clear();

        ptr_desc_db->queryFeatureHamming(dmng_ptr, cand_indx, cand_corr, initial_pose);
        
        if(!cand_indx.empty()){
          int mng_idx = 0;
          while(mng_idx<key_accumsize_vec.size() && cand_indx[0]>key_accumsize_vec[mng_idx]){//cnd_id可能为0
            mng_idx++;
          }
          if(datamng_[sess_id]->robo_type_ == DataManage::RoboType::GROUND && datamng_[mng_idx]->robo_type_ == DataManage::RoboType::GROUND){
            inlier_thre_ = inlier_thre_gg;
            fit_thre_ = fit_thre_gg;
          }
          else{
            inlier_thre_ = inlier_thre_he;
            fit_thre_ = fit_thre_he;
          }
          if(cand_corr[0]>=inlier_thre_){
            printf("%d keycloud have matched %d \n", dmng_ptr->getIntID(), cand_indx[0]);

            if(datamng_[sess_id]->robo_type_ == DataManage::RoboType::GROUND && datamng_[mng_idx]->robo_type_ == DataManage::RoboType::GROUND){
              initial_pose(2,3) = 0.0;
            }

            Eigen::Matrix4d T1 = Eigen::Matrix4d::Identity();
            T1.block<3,3>(0,0) = key_grav_ests[dmng_ptr->getIntID()].toRotationMatrix();  
            Eigen::Matrix4d T2 = Eigen::Matrix4d::Identity();
            T2.block<3,3>(0,0) = key_grav_ests[cand_indx[0]].toRotationMatrix();           
            initial_pose =  T1.inverse()*initial_pose*T2;
            std::cout<<"initial_pose"<<std::endl;
            std::cout<<initial_pose<<std::endl;

            stp.start();
            TicToc clk;
            clk.tic();
            CHECK_EQ(dmng_ptr->getIntID(), keyCloudInd);
            double fitness = fineRegistration(key_cloud_vec_[cand_indx[0]], key_cloud_vec_[keyCloudInd], initial_pose.cast<float>(), final_pose);
            // std::cout<<final_pose<<std::endl;
            std::cout<<"Time fineRegistration "<<clk.toc()<<std::endl;
            stp.record("fineRegistration");

            std::cout << "fineRegistration fitness " << fitness << std::endl;
            if(fitness>0 &&fitness < fit_thre_){ 
              has_loop_flag = true;

              Eigen::Isometry3d tmp_iso;
              tmp_iso.matrix() = final_pose.cast<double>();
              std::cout << "fineRegistration T:\n"
                        << tmp_iso.matrix() << std::endl;

              CHECK_GT(dmng_ptr->getIntID(), cand_indx[0]);
              loop_pairs.emplace_back(std::make_pair(dmng_ptr->getIntID(), cand_indx[0]));
              int global_idx1 = key_idx_vec[loop_pairs.back().first];
              int global_idx2 = key_idx_vec[loop_pairs.back().second];
              Eigen::Isometry3d pose1 = global_odom_poses[global_idx1];//
              if(sess_id > 0) pose1 = adjustPoseZAxis(pose1);
              Eigen::Isometry3d pose2 = global_odom_poses[global_idx2];
              if(sess_id > 0 && cand_indx[0] > key_accumsize_vec[sess_id-1]) pose2 = adjustPoseZAxis(pose2); //if detect self loop
              visualizeloop(pose1, pose2, loop_pairs.size()-1, pubEdge);

              gtsam::Rot3 rotation(tmp_iso.rotation());
              gtsam::Point3 translation(tmp_iso.translation());
              gtsam::Pose3 pose_estimate(rotation, translation);
              robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore,
                                    loopNoiseScore*100, loopNoiseScore*100, loopNoiseScore*100;
              gtsam::noiseModel::Base::shared_ptr robustLoopNoise =
                  gtsam::noiseModel::Robust::Create(
                      gtsam::noiseModel::mEstimator::Cauchy::Create(1),
                      gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));

              graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                      key_idx_vec[dmng_ptr->getIntID()], //relative to this
                      key_idx_vec[cand_indx[0]],//idx of global
                      pose_estimate,
                      robustLoopNoise));
              if(sess_id>0 && key_idx_vec[cand_indx[0]] <= ind_accumsize_vec[sess_id-1]){
                printf("detect inter_loop!!!!\n");
                has_inter_loop++;
                if(has_inter_loop == 1){
                  world_relative_pose = global_odom_poses[key_idx_vec[cand_indx[0]]]*tmp_iso.inverse()*(global_odom_poses[key_idx_vec[dmng_ptr->getIntID()]].inverse());
                  // std::cout<<tmp_iso.matrix()<<std::endl;
                  // std::cout<<world_relative_pose.matrix()<<std::endl;
                  // std::cout<<key_idx_vec[cand_indx[0]]<<" "<<key_odom_poses[cand_indx[0]].matrix()<<std::endl;
                  // std::cout<<key_idx_vec[dmng_ptr->getIntID()]<<" "<<key_odom_poses[dmng_ptr->getIntID()].matrix()<<std::endl;
                }
              }
              else{
                printf("detect intra_loop!!!!\n");
              }
              
            }
          }
        }

        //----!step3. Add descriptors to the database----//
        stp.start();
        ptr_desc_db->addClassToHBST(dmng_ptr);
        stp.record("add to hbst");
        
        // key_cloud_vec_.push_back(key_cloud->makeShared());
        keyCloudInd++;
      }      
      CloudInd++;
      // printf("update %d\n", CloudInd);
      if(sess_id>0 && !has_inter_loop){}
      else{
        //====isam update
        isam.update(graph, initial);
        isam.update();
        if(has_loop_flag){
          isam.update();
          isam.update();
          isam.update();
          // has_loop_flag=false;
        } 
        graph.resize(0);
        initial.clear();
      }
    }
    key_accumsize_vec.emplace_back(keyCloudInd-1);//[0,n][n+1,m]....
    ind_accumsize_vec.emplace_back(CloudInd-1);//[0,n][n+1,m]....
    // for(auto ind:ind_accumsize_vec){
    //   printf("ind_accumsize_vec%d",ind);
    // }
    //reset submap info
    travel_dis = 0.0;
    record_keypose = false;
    local_map_->Clear();
    while(keyCloudInd<key_idx_vec.size()){
      key_odom_poses.pop_back();
      key_grav_ests.pop_back();
      // gravity_est.rotate(rotation_yx);
      key_idx_vec.pop_back(); 
      std::cout<<"pop Redundant for next session"<<std::endl;
    } 
    CHECK_EQ(keyCloudInd, key_idx_vec.size());
    CHECK_EQ(key_grav_ests.size(), key_idx_vec.size());

    //display map result
    if(has_loop_flag){
        // gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial);
        // curr_estimate = optimizer.optimize();

        curr_estimate = isam.calculateEstimate();

        update_poses(curr_estimate, global_odom_poses);
        visualizeGraph(key_idx_vec, global_odom_poses, loop_pairs, key_accumsize_vec, pubEdge, pubKey);
        for(int i = 0; i<key_accumsize_vec.size(); i++){
          std::vector<Eigen::Isometry3d>::iterator odom_begin;
          if(i==0){
            odom_begin = global_odom_poses.begin();
          }
          else{
            odom_begin = global_odom_poses.begin() + ind_accumsize_vec[i-1] + 1;
          }
          std::vector<Eigen::Isometry3d>::iterator odom_end = global_odom_poses.begin() + ind_accumsize_vec[i];
          datamng_[i]->updatePathMsg(odom_begin, odom_end);
        }
        // -----//
        // full map
        full_map->clear();
        for (int i = 0; i < global_odom_poses.size(); ++i) {
          pcl::PointCloud<pcl::PointXYZ> correct_cloud;
          pcl::transformPointCloud(*cloud_vec_[i], correct_cloud, Eigen::Affine3d(global_odom_poses[i]));
          *full_map += correct_cloud;
        }
        voxelDownSample(*full_map, 0.1);

        sensor_msgs::PointCloud2 pub_cloud;
        pcl::toROSMsg(*full_map, pub_cloud);
        pub_cloud.header.frame_id = "camera_init";
        pubCorrectCloud.publish(pub_cloud);

        #if SAVE_MID_FILE
          std::string f_name = PROJ_DIR + "/results/full_map" +
                              std::to_string(keyCloudInd)+ ".pcd";
          pcl::io::savePCDFileASCII (f_name, *full_map);
        #endif
    }
  }

  // human read
  CHECK_EQ(datamng_.size(), key_accumsize_vec.size());
  for(int i=0 ;i<ground_robonum+air_robonum;i++){
    std::cout<<i<<"robo data num "<< datamng_[i]->odom_poses_vec.size()<<" accumkey num"<<key_accumsize_vec[i]<<std::endl;
  }
  stp.printScreen(true);
  const std::string log_dir = PROJ_DIR + "/log/";
  stp.printFile(log_dir + "timing_multi_session_v.txt", true);

  return 0;
}
