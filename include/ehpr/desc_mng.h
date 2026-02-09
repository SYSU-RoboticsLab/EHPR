#ifndef EHPR_DESC_MNG_H
#define EHPR_DESC_MNG_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <bitset>
#include <set>
#include <map>
#include <queue>
#include <unordered_map>
#include <string>
#include <algorithm>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

//// For SIFT:
// #include <opencv2/xfeatures2d.hpp>

#include <boost/dynamic_bitset.hpp>
#include <utility>

#include "tools/bm_util.h"
typedef Eigen::Matrix<float, 2, 1> V2F;
typedef Eigen::Matrix<float, 2, 2> M2F;
typedef Eigen::Matrix<double, 2, 1> V2D;
typedef Eigen::Matrix<double, 2, 2> M2D;

namespace ORB_PARAM{
// fixed parameters for OpenCV ORB Features
  static constexpr float scale_factor = 1.00;
  static constexpr int n_levels = 1;
  static constexpr int first_level = 0;
  static constexpr int WTA_K = 2;
  static constexpr int nfeatures = 250;
  static constexpr int edge_threshold = 31;
  static constexpr int score_type = 0;
  static constexpr int patch_size = 31;
  static constexpr int fast_threshold = 20;
} 
namespace SIFT_PARAM{
// fixed parameters for OpenCV SIFT Features
  static constexpr int nfeatures = 500; //0
  static constexpr int nOctaveLayers = 3;//3
  static constexpr double contrastThreshold = 0.01;//0.04
  static constexpr double edgeThreshold = 20;//10
  static constexpr double sigma = 1.1;//1.6
} 

struct DescriptorManagerConfig {
  float reso_row_ = 1.0f, reso_col_ = 1.0f;
  int n_row_ = 120, n_col_ = 120;
  double max_dens_ = 255;
  
  float height_res_ = 1.0f;
  float lidar_height_ = 2.0f;  // ground assumption
  float blind_sq_ = 9.0f;

  int local_density_cols_ = 1;
  int local_density_rows_ = 1;
};

union Pixelf {
  struct {
    float row_f;
    float col_f;
    float elev;
  };
  int data[3]{};

  Pixelf(float r, float c, float e) : row_f(r), col_f(c), elev(e) {}

  Pixelf() {
    row_f = -1;
    col_f = -1;
    elev = -1;
  }

  bool operator<(const Pixelf &b) const {
    return row_f < b.row_f;
  }
};

//----manage the collection of descriptors or feature in a submap---//
class DescriptorManager {
  // configuration
  const DescriptorManagerConfig cfg_;
  const float VAL_ABS_INF_ = 1e3;

  // property
  float x_max_, x_min_, y_max_, y_min_;  // for points in the sensor frame, not in the bev frame
  std::string str_id_;
  int int_id_;

  float h_max_, h_min_;

  // data
  std::vector<std::vector<boost::dynamic_bitset<>>> height_bin_mat_;
  std::vector<Eigen::Vector2d> keypoints_; 
  std::vector<boost::dynamic_bitset<>> key_height_bin_; 
  int height_bin_num_;

  cv::Mat1f bev_;
  
  std::vector<std::pair<int, Pixelf>> bev_pixfs_; // float row col height, and the hash generated from discrete row column.
  float max_bin_val_ = -VAL_ABS_INF_, min_bin_val_ = VAL_ABS_INF_;

  Eigen::Isometry3d local_pose_;
  Eigen::Quaterniond local_grav_;

protected:
  template<typename PointType>
  ///
  /// \tparam PointType
  /// \param pt
  /// \return (row, col) of the matched point
  std::pair<int, int> hashPointToImage(const PointType &pt) const {
    std::pair<int, int> res{-1, -1};
    float padding = 1e-2;
    if (pt.x < x_min_ + padding || pt.x > x_max_ - padding || pt.y < y_min_ + padding || pt.y > y_max_ - padding) {
//      std::cout << pt.x << "\t" << pt.y << std::endl;
      return res;
    }
    res.first = int(std::floor(pt.x / cfg_.reso_row_)) + cfg_.n_row_ / 2;
    res.second = int(std::floor(pt.y / cfg_.reso_col_)) + cfg_.n_col_ / 2;

    DCHECK(res.first >= 0 && res.first < cfg_.n_row_);
    DCHECK(res.second >= 0 && res.second < cfg_.n_col_);

    return res;
  }

  /// transform points from the lidar frame to continuous image frame
  /// \param p_in_l
  /// \return
  V2F pointToContRowCol(const V2F &p_in_l) const {
    V2F continuous_rc(p_in_l.x() / cfg_.reso_row_ + cfg_.n_row_ / 2 - 0.5f,
                      p_in_l.y() / cfg_.reso_col_ + cfg_.n_col_ / 2 - 0.5f);
    return continuous_rc;
  }

public:
  int attach_idx_; //there for dataset seq
  cv::Mat bev_descriptors_; // descriptors
  explicit DescriptorManager(const DescriptorManagerConfig &config, int int_id, 
                          const Eigen::Isometry3d &transform = Eigen::Isometry3d::Identity(), 
                          const Eigen::Quaterniond &gra_dir = Eigen::Quaterniond::Identity()) : cfg_(config), int_id_(int_id) {
    CHECK(cfg_.n_col_ % 2 == 0);
    CHECK(cfg_.n_row_ % 2 == 0);

    x_min_ = -(cfg_.n_row_ / 2) * cfg_.reso_row_;
    x_max_ = -x_min_;
    y_min_ = -(cfg_.n_col_ / 2) * cfg_.reso_col_;
    y_max_ = -y_min_;

    bev_ = cv::Mat::zeros(cfg_.n_row_, cfg_.n_col_, CV_32F) * (-VAL_ABS_INF_);//mr test
    height_bin_mat_.resize(cfg_.n_row_ , std::vector<boost::dynamic_bitset<>>(cfg_.n_col_));

    attach_idx_ = int_id;
    local_pose_ = transform;
    local_grav_ = gra_dir;
  }

  template<typename PointType>
  void makeDensityBEV(typename pcl::PointCloud<PointType>::ConstPtr &ptr_gapc, std::string str_id = "", 
                      const Eigen::Vector3d max_xyz = Eigen::Vector3d::Zero(),
                      const Eigen::Vector3d min_xyz = Eigen::Vector3d::Zero(),
                      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_extra = nullptr) {  //
    CHECK(ptr_gapc);
    CHECK_GT(ptr_gapc->size(), 10);
    TicToc clk;
    clk.tic();

    float max_g_h = -1000, min_g_h =  1000;
    if(max_xyz[2] <= min_xyz[2] ){
      if(cloud_extra != nullptr){
        for (const auto &pt: cloud_extra->points) {
          float height = pt.z;
          max_g_h = std::max(max_g_h , height);
          min_g_h = std::min(min_g_h , height);
        }
      }
      for (const auto &pt: ptr_gapc->points) {
        float height = pt.z;
        max_g_h = std::max(max_g_h , height);
        min_g_h = std::min(min_g_h , height);     
      }
    }
    else{
      std::cout<<"already have height max min"<<std::endl;
      max_g_h = max_xyz[2];
      min_g_h = min_xyz[2]; 
    }

    //prepare for height coding
    CHECK_GT(max_g_h, min_g_h);
    printf("Max/Min cloud height: %f %f\n", max_g_h, min_g_h);  
    float height_bin_base = min_g_h - cfg_.height_res_/2;
    height_bin_num_ = ceil((max_g_h - height_bin_base)/cfg_.height_res_);
    h_max_ = max_g_h; 
    h_min_ = height_bin_base; 

    //record intensity and height coding
    bev_ = cv::Mat::zeros(cfg_.n_row_, cfg_.n_col_, CV_32F);
    float max_bin_val_ = -VAL_ABS_INF_, min_bin_val_ = VAL_ABS_INF_;

    int cell_count = 0;
    std::map<int, Pixelf> tmp_pillars;
    for (const auto &pt: ptr_gapc->points) {
      std::pair<int, int> rc = hashPointToImage<PointType>(pt);
      if (rc.first > 0) {
        bev_(rc.first, rc.second) += 1;

        if(bev_(rc.first, rc.second) >= cfg_.max_dens_)// 255
        bev_(rc.first, rc.second) = cfg_.max_dens_;

        max_bin_val_ = max_bin_val_ < bev_(rc.first, rc.second) ? bev_(rc.first, rc.second) : max_bin_val_;
        min_bin_val_ = min_bin_val_ > bev_(rc.first, rc.second) ? bev_(rc.first, rc.second) : min_bin_val_;

        V2F coor_f = pointToContRowCol(V2F(pt.x, pt.y));
        tmp_pillars[rc.first * cfg_.n_col_ + rc.second] = Pixelf(coor_f.x(), coor_f.y(),  bev_(rc.first, rc.second));

        // //height coding
        float height = pt.z;
        auto &tmp_bin = height_bin_mat_[rc.first][rc.second];
        if(tmp_bin.size() != height_bin_num_){
          tmp_bin.resize(height_bin_num_);
        }
        int idx = floor((height - height_bin_base)/cfg_.height_res_);
        tmp_bin.set(idx, true); 
      }
    }

    if(cloud_extra != nullptr){
      for (const auto &pt: cloud_extra->points) {
        std::pair<int, int> rc = hashPointToImage<PointType>(pt);
        if (rc.first > 0) {
          float height = pt.z;
          auto &tmp_bin = height_bin_mat_[rc.first][rc.second];
          if(tmp_bin.size() !=height_bin_num_){
            tmp_bin.resize(height_bin_num_);
          }
          int idx = floor((height - height_bin_base)/cfg_.height_res_);
          // std::cout << "set bin  " << idx << std::endl;
          tmp_bin.set(idx, true);
        }
      }
    }

    bev_pixfs_.clear();
    bev_pixfs_.insert(bev_pixfs_.begin(), tmp_pillars.begin(), tmp_pillars.end());

    std::cout << "Time make density bev and height: " << clk.toc() << std::endl;

    if (!str_id.empty())
      str_id_ = std::move(str_id);
    else
      str_id_ = std::to_string(ptr_gapc->header.stamp);

    // max_bin_val_ = *std::max_element(bev_.begin(), bev_.end());
    // min_bin_val_ = *std::min_element(bev_.begin(), bev_.end());

    cv::Mat mask, view;
    view = cv::Mat::zeros(bev_.size(), bev_.type());
    inRange(bev_, cv::Scalar::all(max_bin_val_* 0.05), cv::Scalar::all(max_bin_val_), mask);  //300 and 0,05 in mulran is good
    normalize(bev_, view, 1, 255, cv::NORM_MINMAX, -1, mask);
    printf("Max/Min bin density: %f %f\n", max_bin_val_, min_bin_val_);

    bev_ = view;
#if SAVE_MID_FILE
    cv::Mat invert_mat = cv::Mat::ones(view.size(), view.type()) * 255;
    // invert_mat = invert_mat - view;
    invert_mat = view;
    std::string dir = std::string(PJSRCDIR) + "/results/bev_img/";
    cv::imwrite(dir + "bev_img-" + str_id_ + ".png", invert_mat);
#endif
  }

  void makeFeature();

  void clearImage(){
    bev_.release();
    bev_descriptors_.release();

    decltype(height_bin_mat_)().swap(height_bin_mat_);
  }

  cv::Mat1f getBevImage() const {
    if (bev_.empty()) {
      cv::Mat1f tmp = cv::Mat::ones(cfg_.n_row_, cfg_.n_col_, CV_32F) * (-VAL_ABS_INF_);
      for (const auto &pillar: bev_pixfs_) {
        int rr = pillar.first / cfg_.n_col_;
        int cc = pillar.first % cfg_.n_col_;
        DCHECK_LT(rr, cfg_.n_row_);
        DCHECK_LT(cc, cfg_.n_col_);
        tmp(rr, cc) = pillar.second.elev;
      }
      return tmp;
    } else
      return bev_.clone();
  }


  const Eigen::Isometry3d &getAnchorPose() const {
    return local_pose_;
  }

  const Eigen::Vector2d &getKeyPoint(int seq) const {
    return keypoints_[seq];
  }

  const std::vector<Eigen::Vector2d> &getKeyPointVec() const {
    return keypoints_;
  }

  const int getLevKeyPointsSize() const {
    return keypoints_.size();
  }

  const boost::dynamic_bitset<> &getKeyHeightBin(int seq) const {
    return key_height_bin_[seq];
  }

  const std::pair<float, float> getHightMinMax()const {
    return std::make_pair(h_max_, h_min_);
  }

  void getBinInfo(int &num, float &res)const {
    num = height_bin_num_;
    res = cfg_.height_res_;
  }

  inline std::string getStrID() const {
    return str_id_;
  }

  inline int getIntID() const {
    return int_id_;
  }

  inline const DescriptorManagerConfig &getConfig() const {
    return cfg_;
  }

};


#endif //EHPR_DESC_MNG_H