#ifndef DATA_TOOLS_ROS_H
#define DATA_TOOLS_ROS_H

#include <utility>
#include <vector>
#include <iostream>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <glog/logging.h>
#include "tools/algos.h"

template<typename PointType>
typename pcl::PointCloud<PointType>::Ptr readKITTIPointCloudBin(const std::string &lidar_bin_path) {
  typename pcl::PointCloud<PointType>::Ptr out_ptr = nullptr;

  // allocate 4 MB buffer (only ~130*4*4 KB are needed)
  int num = 1000000;
  auto *data = (float *) malloc(num * sizeof(float));
  // pointers
  float *px = data + 0;
  float *py = data + 1;
  float *pz = data + 2;
  float *pr = data + 3;

  FILE *stream;
  stream = fopen(lidar_bin_path.c_str(), "rb");
  if (stream) {
    num = fread(data, sizeof(float), num, stream) / 4;
    out_ptr.reset(new pcl::PointCloud<PointType>());
    out_ptr->reserve(num);
    for (int32_t i = 0; i < num; i++) {
      PointType pt;
      pt.x = *px;
      pt.y = *py;
      pt.z = *pz;
      out_ptr->push_back(pt);

      px += 4;
      py += 4;
      pz += 4;
      pr += 4;
    }
    fclose(stream);

  } else {
    printf("Lidar bin file %s does not exist.\n", lidar_bin_path.c_str());
    exit(-1);
  }
  free(data);
  return out_ptr;
}

template<int dim>
struct SimpleRMSE {
  double sum_sqs = 0;
  double sum_abs = 0;
  int cnt_sqs = 0;

  SimpleRMSE() = default;

  void addOneErr(const double *d) {
    cnt_sqs++;
    double tmp = 0;
    for (int i = 0; i < dim; i++) {
      tmp += d[i] * d[i];
    }
    sum_sqs += tmp;
    sum_abs += std::sqrt(tmp);
  }

  double getRMSE() const { return cnt_sqs ? std::sqrt(sum_sqs / cnt_sqs) : -1; }

  double getMean() const { return cnt_sqs ? sum_abs / cnt_sqs : -1; }
};

struct PredictionOutcome {
  enum Res {
    TP, FP, TN, FN
  };

  int id_src = -1;  //
  int id_tgt = -1;
  Res tfpn = Res::TN; // the most insignificant type
  double est_err[4]{};  // TP, FP: the error param on SE2, else: all zero
  double correlation{};
  int match_count = 0;
  double match_ratio = 0.0;
  double attach_info[4]{}; 
};

// Definition of loader
// 1. use script to generate 2 files: (all ts are in ns)
//  1) timestamp and gt pose (z up) of the sensor. Ordered by gt ts. (13 elements per line)
//  2) timestamp, seq, and the path (no space) of each lidar scan bin file.
//    Ordered by lidar ts AND seq. (3 elements per line)
// 2. load the .bin data in sequence, and find the gt pose
class Datatools {
  struct LaserScanInfo {
    bool has_gt_positive_lc = false;
    Eigen::Isometry3d sens_pose;
    Eigen::Quaterniond grav_est;
    int seq = 0;
    double ts = 0;
    std::string fpath;

    LaserScanInfo() = default;
  };

  // valid input info about lidar scans

  std::vector<int> assigned_seqs_;  // actually a seq:addr map
  
  // param:
  const double ts_diff_tol = 0.01;  // 10ms. In Mulran, the gt is given at an interval of about 10ms
  const double min_time_excl = 15.0;  // exclude 15s
  const double sim_thres = 0.64928;  // the similarity score when determining TFPN
  const int min_seq_excl = 5;
  const double positive_dis_thres = 70.0;

  // bookkeeping variables
  int p_lidar_curr = -1;
//  std::map<int, LaserScanInfo>::iterator it_lidar_curr;  // int index is a little safer than iterator/pointer?

  // benchmark recorders
  SimpleRMSE<2> tp_trans_rmse, all_trans_rmse;
  SimpleRMSE<1> tp_rot_rmse, all_rot_rmse;
  std::vector<PredictionOutcome> pred_records;

public:
  std::vector<LaserScanInfo> laser_info_;  // use predefined {seq-id, bin file} for better traceability
  std::vector<LaserScanInfo> laser_info_qr_; 
  
  Datatools(){
    laser_info_.clear();
    laser_info_qr_.clear();
  }

  static void savePointCloudToBin(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    for (const auto& point : cloud->points) {
        ofs.write(reinterpret_cast<const char*>(&point.x), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&point.y), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&point.z), sizeof(float));
    }
    ofs.close();
  }
  static void loadClouds(std::string fpath_laser, std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>&clouds_vec, const std::vector<Eigen::Isometry3d> &poses_vec, std::string rootpath=""){
    std::fstream infile2;
    std::string sbuf;
    infile2.open(fpath_laser, std::ios::in);
    if (infile2.rdstate() != std::ifstream::goodbit) {
      std::cerr << "Error opening laser info file: " << fpath_laser << std::endl;
      return;
    }

    std::vector<double> lidar_ts;
    std::vector<int> assigned_seq;
    while (std::getline(infile2, sbuf)) {
      std::istringstream iss(sbuf);

      double ts;
      int seq;
      std::string bin_path;

      if (iss >> ts) {
        lidar_ts.emplace_back(ts);

        iss >> seq;
        assigned_seq.emplace_back(seq);

        iss >> bin_path;

        std::string lidar_fpath = std::string(rootpath+bin_path);
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_ptr(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::io::loadPCDFile<pcl::PointXYZ>(lidar_fpath, *input_ptr);
        clouds_vec.emplace_back(input_ptr);
      }
    }
    infile2.close();
    CHECK_EQ(lidar_ts.size(), assigned_seq.size());
    CHECK_EQ(lidar_ts.size(), clouds_vec.size());
    printf("Added %lu laser bin paths.\n", clouds_vec.size());

  }

  static void loadCloudsPath(std::string fpath_laser, std::vector<std::string>&cloud_path_vec, const std::vector<Eigen::Isometry3d> &poses_vec, std::string rootpath=""){
    std::fstream infile2;
    std::string sbuf;
    infile2.open(fpath_laser, std::ios::in);
    if (infile2.rdstate() != std::ifstream::goodbit) {
      std::cerr << "Error opening laser info file: " << fpath_laser << std::endl;
      return;
    }

    std::vector<double> lidar_ts;
    std::vector<int> assigned_seq;
    while (std::getline(infile2, sbuf)) {
      std::istringstream iss(sbuf);

      double ts;
      int seq;
      std::string bin_path;

      if (iss >> ts) {
        lidar_ts.emplace_back(ts);

        iss >> seq;
        assigned_seq.emplace_back(seq);

        iss >> bin_path;

        std::string lidar_fpath = std::string(rootpath+bin_path);

        cloud_path_vec.emplace_back(lidar_fpath);
      }
    }
    infile2.close();
    CHECK_EQ(lidar_ts.size(), poses_vec.size());
    CHECK_EQ(lidar_ts.size(), cloud_path_vec.size());
    printf("Added %lu laser bin paths.\n", cloud_path_vec.size());

  }

  static void loadIMUs(const std::string &imu_filepath, std::vector<Eigen::Quaterniond> &imu_orientations, std::vector<double> &imu_times) {
    std::ifstream file(imu_filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open IMU data file.");
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        double time, x, y, z, w;

        // Assuming the file format is: time, orientation.x, orientation.y, orientation.z, orientation.w, ...
        ss >> time >> x >> y >> z >> w;

        // Store the timestamp
        imu_times.push_back(time);

        // Store the orientation as a quaternion
        Eigen::Quaterniond orientation(w, x, y, z);
        imu_orientations.push_back(orientation);
    }
    file.close();
    printf("Added %lu stamped imu info.\n", imu_orientations.size());
  }

  static void loadPoses(std::string fpath_pose, std::vector<Eigen::Isometry3d>&poses_vec, std::vector<double> &times_vec){
    std::fstream infile1;
    std::string sbuf, pname;

    infile1.open(fpath_pose, std::ios::in);
    if (infile1.rdstate() != std::ifstream::goodbit) {
      std::cerr << "Error opening gt pose file: " << fpath_pose << std::endl;
      return;
    }

    const int line_len = 13; // timestamp and the 12-element sensor gt pose
    while (std::getline(infile1, sbuf)) {
      std::istringstream iss(sbuf);

      double tmp;
      Eigen::Vector3d tmp_trans;
      Eigen::Matrix3d tmp_rot_mat;
      Eigen::Quaterniond tmp_rot_q;
      Eigen::Isometry3d tmp_tf;
      for (int i = 0; i < line_len; i++) {
        CHECK(iss >> tmp);
        if (i == 0)
          times_vec.push_back(tmp);
        else {
          if ((i - 1) % 4 == 3)
            tmp_trans((i - 1) / 4) = tmp;
          else
//            tmp_rot_mat((i - 1) / 4, (i - 1) % 4);  // `-Wuninitialized` cannot detect this bug!
            tmp_rot_mat((i - 1) / 4, (i - 1) % 4) = tmp;
        }
      }
      tmp_rot_q = Eigen::Quaterniond(tmp_rot_mat);
      tmp_tf.setIdentity();
      tmp_tf.rotate(tmp_rot_q);
      tmp_tf.pretranslate(tmp_trans);

      poses_vec.emplace_back(tmp_tf);
    }
    infile1.close();
    CHECK_EQ(poses_vec.size(), times_vec.size());
    printf("Added %lu stamped gt poses.\n", poses_vec.size());
    
    // for time sort
    // std::vector<int> sort_permu(poses_vec.size());
    // std::iota(sort_permu.begin(), sort_permu.end(), 0);
    // std::sort(sort_permu.begin(), sort_permu.end(), [&](const int &a, const int &b) {
    //   return times_vec[a] < times_vec[b];
    // });
    // apply_sort_permutation(sort_permu, poses_vec);
    // apply_sort_permutation(sort_permu, times_vec);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr loadNewScan(int index) {
    // Load the scan into the cache so that it can be retrieved by calling related functions
    CHECK(index >= 0);
    if (index >= laser_info_.size()) {
      printf("\n===\ncurrent addr %d exceeds boundary\n", index);
      return nullptr;
    }
    printf("\n===\nloaded scan addr %d, seq: %d, fpath: %s\n", index, laser_info_[index].seq,
           laser_info_[index].fpath.c_str());
    // pcl::PointCloud<pcl::PointXYZ>::Ptr out_ptr = readKITTIPointCloudBin<pcl::PointXYZ>(laser_info_[index].fpath);

    pcl::PointCloud<pcl::PointXYZ>::Ptr input_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile<pcl::PointXYZ>(laser_info_[index].fpath, *input_ptr);
    pcl::PointCloud<pcl::PointXYZ>::Ptr out_ptr = input_ptr;

    return out_ptr;
  }

  // 1. loader
  // 3. Result saver
  void savePredictionResults(const std::string &sav_path) const {
    std::fstream res_file(sav_path, std::ios::out);

    if (res_file.rdstate() != std::ifstream::goodbit) {
      std::cerr << "Error opening " << sav_path << std::endl;
      return;
    }

    // tgt before src
    for (const auto &rec: pred_records) {
      int addr_tgt = lookupNN<int>(rec.id_tgt, assigned_seqs_, 0);
      CHECK_GE(addr_tgt, 0);

      res_file << rec.tfpn << "\t";

      std::string str_rep_tgt = laser_info_[addr_tgt].fpath, str_rep_src;

      if (rec.id_src < 0) {
        res_file << rec.id_tgt << "-x" << "\t";
        str_rep_src = "x";
      } else {
        int addr_src = lookupNN<int>(rec.id_src, assigned_seqs_, 0);
        CHECK_GE(addr_src, 0);

        res_file << rec.id_tgt << "-" << rec.id_src << "\t";
        str_rep_src = laser_info_[addr_src].fpath;
      }

      res_file << rec.correlation << "\t" << rec.est_err[0] << "\t" << rec.est_err[1] << "\t" << rec.est_err[2] << "\t" << rec.est_err[3] << "\t";
      res_file << rec.match_count << "\t" << rec.match_ratio << "\t";
//      // case 1: path
//      res_file << str_rep_tgt << "\t" << str_rep_src << "\n"; // may be too long

      // case 2: shortened
      int str_max_len = 32;
      int beg_tgt = std::max(0, (int) str_rep_tgt.length() - str_max_len);
      int beg_src = std::max(0, (int) str_rep_src.length() - str_max_len);
      res_file << str_rep_tgt.substr(beg_tgt, str_rep_tgt.length() - beg_tgt) << "\t"
               << str_rep_src.substr(beg_src, str_rep_src.length() - beg_src) << "\n";
      
    }
    // rmse and mean error can be calculated from this file. So we will not record it.

    printf("In outcome file:\n");
    printf("TP is %d\n", static_cast<std::underlying_type<PredictionOutcome::Res>::type>(PredictionOutcome::TP));
    printf("FP is %d\n", static_cast<std::underlying_type<PredictionOutcome::Res>::type>(PredictionOutcome::FP));
    printf("TN is %d\n", static_cast<std::underlying_type<PredictionOutcome::Res>::type>(PredictionOutcome::TN));
    printf("FN is %d\n", static_cast<std::underlying_type<PredictionOutcome::Res>::type>(PredictionOutcome::FN));

    res_file.close();
    printf("Outcome saved successfully.\n");

  }

  PredictionOutcome
  addPredictions(const int query_index, const std::vector<int> &cand_index , const std::vector<double> &cand_cor,
                const Eigen::Matrix4d &T_est_delta_3d = Eigen::Matrix4d::Identity(),
                const int & match = 0,
                const double & match_ratio = 0.0) {
    
    PredictionOutcome curr_res;
    curr_res.id_tgt = laser_info_[query_index].seq;
    
    if(!cand_index.empty()){
      curr_res.id_src = laser_info_[cand_index[0]].seq;
      curr_res.correlation = cand_cor[0];
      
      //estimate error
      // double err_vec[3] = {tf_err.translation().x(), tf_err.translation().y(), std::atan2(tf_err(1, 0), tf_err(0, 0))};
      // printf(" Error: dx=%f, dy=%f, dtheta=%f\n", err_vec[0], err_vec[1], err_vec[2]);
      double gt_dist = (laser_info_[query_index].sens_pose.translation() - laser_info_[cand_index[0]].sens_pose.translation()).head<2>().norm();
      if(gt_dist < positive_dis_thres){
        curr_res.tfpn = PredictionOutcome::TP;
      }
      else{
        curr_res.tfpn = PredictionOutcome::FP;
      }
    }
    else{
      if (laser_info_[query_index].has_gt_positive_lc)  // FN
        curr_res.tfpn = PredictionOutcome::FN;
      else  // TN
        curr_res.tfpn = PredictionOutcome::TN;
    }
    curr_res.match_count = match;
    curr_res.match_ratio = match_ratio;
    pred_records.emplace_back(curr_res);
    return curr_res;
  }

  void computeGT(){
    int cnt_gt_lc_p = 0;
    int cnt_gt_lc = 0;
    for (auto &it_fast: laser_info_qr_) {  // not necessarily ordered by assigned seq
      for (auto &it_slow: laser_info_) {
        double dist = (it_fast.sens_pose.translation() - it_slow.sens_pose.translation()).head<2>().norm();
        if (dist < positive_dis_thres) {
          if (!it_fast.has_gt_positive_lc) {
            it_fast.has_gt_positive_lc = true;
            cnt_gt_lc_p++;//一个可能配一个计数
          }
          cnt_gt_lc++; //一个可能配准多个计数
        }
      }
    }
  }

  static Eigen::Isometry3d evalMetricEst(const Eigen::Isometry3d &T_delta, const Eigen::Isometry3d &gt_src_3d,
                                         const Eigen::Isometry3d &gt_tgt_3d) {

    Eigen::Isometry3d T_relative_gt = gt_tgt_3d.inverse() * gt_src_3d;

    // TODO: project gt 3d into some gt 2d, and use
    Eigen::Isometry2d T_tsen_ssen2_gt;
    T_tsen_ssen2_gt.setIdentity();
    // for translation: just the xy difference
    // for rotation: rotate so that the two z axis align
    Eigen::Vector3d z0(0, 0, 1);
    Eigen::Vector3d z1 = T_relative_gt.matrix().block<3, 1>(0, 2);
    Eigen::Vector3d ax = z0.cross(z1).normalized();
    double ang = acos(z0.dot(z1));
    Eigen::AngleAxisd d_rot(-ang, ax);

    Eigen::Isometry3d T_relative_recti = d_rot * T_relative_gt;  // only top 2x2 useful

    // T_tsen_ssen2_gt.rotate(std::atan2(R_rectified(1, 0), R_rectified(0, 0)));
    // T_tsen_ssen2_gt.pretranslate(Eigen::Vector2d(T_relative_gt.translation().segment(0, 2)));  // only xy

    // std::cout << "T delta gt 2d:\n" << T_tsen_ssen2_gt.matrix() << std::endl;  // Note T_delta is not comparable to this

    Eigen::Isometry3d T_gt_est = T_relative_recti.inverse() * T_delta;
    return T_gt_est;
  }

  inline double getTPMeanTrans() const { return tp_trans_rmse.getMean(); }

  inline double getTPMeanRot() const { return tp_rot_rmse.getMean(); }

  inline double getTPRMSETrans() const { return tp_trans_rmse.getRMSE(); }

  inline double getTPRMSERot() const { return tp_rot_rmse.getRMSE(); }


};

#endif
