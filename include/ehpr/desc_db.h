#ifndef EHPR_DESC_DB_H
#define EHPR_DESC_DB_H

#include <memory>
#include <algorithm>
#include <set>
#include <unordered_set>

#include <boost/math/distributions/normal.hpp>

#include "tools/bm_util.h"
#include "data_tools.hpp"

#include "ehpr/desc_mng.h"
#include "ehpr/AlignRansac2D.h"
#include "srrg_hbst/types/binary_tree.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
extern SequentialTimeProfiler stp;

static constexpr int descriptor_size_bits = 256;//256
// using Matchable = srrg_hbst::BinaryMatchable<cv::KeyPoint, descriptor_size_bits>;  
// pair<local index,  position>
using Matchable = srrg_hbst::BinaryMatchable<std::pair<int, Eigen::Vector2d>, descriptor_size_bits>;
using Node = srrg_hbst::BinaryNode<Matchable>;
using Tree = srrg_hbst::BinaryTree<Node>;

struct DescriptorDBConfig {
  float reso_row_ = 1.0f, reso_col_ = 1.0f;
  int n_row_ = 120, n_col_ = 120;

  int bucket_size_ = 1;
  int dist_thres_ = 50;
  double distri_mean_ =115.0;
  double distri_var_ = 22.0;
  int skip_num_ = 10;
};

inline int find_last(const boost::dynamic_bitset<>& bitset) {  
  for (int i = bitset.size() - 1; i >= 0; --i) {
      if (bitset[i]) {
          return i;
      }
  }
  return -1; //no find
}

struct HammingBucket {
  std::unique_ptr<Tree> hbst_tree;
  int min_ones; // Left boundary of the interval
  int max_ones; // right boundary of the interval
  int size = 0;
  HammingBucket(int min, int max) : min_ones(min), max_ones(max) {
    hbst_tree = std::make_unique<Tree>();
  }
};

struct LocalCandidate {
  int source_id = -1;
  int target_id = -1;
  size_t number_of_inliers = 0;
  std::vector<std::pair<int, int>> inlier_vec;
};

struct ClosureCandidate {
  int source_id = -1;
  int target_id = -1;
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  size_t number_of_inliers = 0;
  std::vector<int> inlier_vec;
};

class DescriptorDB {
  
  const DescriptorDBConfig cfg_;
  
  std::vector<int> class_size_; // = std::vector<int>(descriptor_size_bits, 0);
  std::vector<HammingBucket> buckets_;
  std::vector<int> count_vec_; 
  // std::unique_ptr<Tree> hbst_binary_tree_ = std::make_unique<Tree>();
  std::vector<Tree::MatchableVector> class_matchables_query_;
  Tree::MatchVectorMap descriptor_matches_;

  std::vector<std::shared_ptr<const DescriptorManager>> all_bevs_;
  std::unordered_map<int, std::vector<map_closures::PointPair>> cand_correspond_;
  std::vector<Eigen::Isometry3d> db_gt_vec;
  std::vector<Eigen::Isometry3d> qr_gt_vec;

  std::vector<Eigen::Quaterniond> db_imu_vec;
  std::vector<Eigen::Quaterniond> qr_imu_vec;
  //for test
public:
  double align_time;

  DescriptorDB(const DescriptorDBConfig &config) : cfg_(config) {
    initializeBuckets();
    class_size_.resize(descriptor_size_bits+1, 0);
    class_matchables_query_.resize(descriptor_size_bits+1);
  }

  void printMatchables(){
    printf("bucket size[%d]: ", buckets_.size());
    for (int i = 0; i < buckets_.size(); i++) {
      printf("%10lu", buckets_[i].size);
    }
    printf("\n"); 
  }
  
  const std::vector<int>& getClassSize(){
    return class_size_;
  }

  size_t getDBSize(){
    return all_bevs_.size();
  }
  
  std::shared_ptr<const DescriptorManager> getKeyFramebyIdx(int index){
    if (index < 0) {
      index += all_bevs_.size(); 
    }
    if (index >= 0 && index < all_bevs_.size()) {
      return all_bevs_[index];
    }
    else{
      LOG(WARNING)<<"wrong index for database";
    }
  }

  void updatePosVec(std::vector<Eigen::Isometry3d>db_vec, std::vector<Eigen::Isometry3d>qr_vec){
    db_gt_vec = db_vec;
    qr_gt_vec = qr_vec;
  }

  void updateGravVec(const std::unique_ptr<Datatools>& data_ptr){
    db_imu_vec.clear();
    qr_imu_vec.clear();
    for(auto& info: data_ptr->laser_info_){
      db_imu_vec.emplace_back(info.grav_est);
    }
    for(auto& info: data_ptr->laser_info_qr_){
      qr_imu_vec.emplace_back(info.grav_est);
    }
  }

  double calcauInlierRatio(int qr_id);

  int queryFeatureHamming(const std::shared_ptr<const DescriptorManager> &q_ptr,
                    std::vector<int> &cand_indx, std::vector<double> &cand_corr,
                    Eigen::Matrix4d &pose);

  
  void initializeBuckets() {
    int num_buckets = cfg_.bucket_size_;
    int max_bits = descriptor_size_bits; 
    
    //--even bucket---//
    // int step = ceil(max_bits / num_buckets);
    // for (int i = 0; i < num_buckets; i++) {//if 0?
    //   int min_ones = i * step + 1;
    //   int max_ones = (i + 1) * step;
    //   buckets_.emplace_back(min_ones, max_ones);
    // }
    // buckets_[0].min_ones = 0;
    // buckets_[num_buckets-1].min_ones = max_bits;
    //--even_bucket---//

    //--nomal bucket---//
    double mean = cfg_.distri_mean_;
    double std_dev = cfg_.distri_var_;
    boost::math::normal_distribution<double> distribution(mean, std::max(std_dev, 1e-9)); // 防止std_dev为0
    std::vector<int> intervals;
    double quantile_value;
    intervals.push_back(-1);
    for (int i = 1; i < num_buckets; ++i) {
        double probability = static_cast<double>(i) / num_buckets; // 计算分位点
        try {
            quantile_value = boost::math::quantile(distribution, probability);
        } catch (const std::exception &e) {
            std::cerr << "Error calculating quantile for probability " << probability << ": " << e.what() << std::endl;
            quantile_value = quantile_value + 1; 
        }
        int interval_point = std::round(quantile_value); //std::max(0, std::min(255, static_cast<int>(quantile_value + 0.5)));
        intervals.push_back(interval_point);
    }
    intervals.push_back(max_bits);

    int offset = 0;
    for (size_t i = 0; i < intervals.size() - 1; ++i) {
      int min_ones = intervals[i] + offset + 1;
      std::cout <<offset<<" ";
      offset = 0;
      int max_ones = intervals[i+1];
      while(min_ones > max_ones){
          offset++;
          max_ones++;
      }
      buckets_.emplace_back(min_ones, max_ones);
    }
    std::cout << std::endl;
    //--nomal bucket---//
  }

  ClosureCandidate ValidateClosure(const int reference_id, const int query_id) const {
    const Tree::MatchVector &matches = descriptor_matches_.at(reference_id);
    const size_t num_matches = matches.size();

    ClosureCandidate closure;
    if (num_matches > 2) {
        // const auto &ref_map_lower_bound = density_maps_.at(reference_id).lower_bound;
        // const auto &qry_map_lower_bound = density_maps_.at(query_id).lower_bound;
        // auto to_world_point = [](const auto &p, const auto &offset) {
        //     return Eigen::Vector2d(p.y + offset.x(), p.x + offset.y());
        // };
        std::vector<map_closures::PointPair> keypoint_pairs(num_matches);
        std::vector<double> distances(num_matches);
        std::transform(
            matches.cbegin(), matches.cend(), keypoint_pairs.begin(),
            [&](const Tree::Match&match) {
                Eigen::Vector2d ref_point = Eigen::Vector2d((match.object_references[0].second.x()-cfg_.n_row_/2)*cfg_.reso_row_, 
                                                            (match.object_references[0].second.y()-cfg_.n_col_/2) * cfg_.reso_col_);
                Eigen::Vector2d query_point = Eigen::Vector2d((match.object_query.second.x()-cfg_.n_row_/2)*cfg_.reso_row_, 
                                                              (match.object_query.second.y()-cfg_.n_col_/2)  * cfg_.reso_col_);

                return map_closures::PointPair(ref_point, query_point, match.distance);
            });

        // std::sort(keypoint_pairs.begin(), keypoint_pairs.end(),
        //   [](const map_closures::PointPair &a, const map_closures::PointPair &b) {
        //       return a.dist < b.dist;
        //   });

        const auto &[pose2d, inliers_vec] = RansacAlignment2D(keypoint_pairs);
        closure.source_id = reference_id;
        closure.target_id = query_id;
        closure.pose.block<2, 2>(0, 0) = pose2d.linear();
        closure.pose.block<2, 1>(0, 3) = pose2d.translation();
        closure.number_of_inliers = inliers_vec.size();
        closure.inlier_vec = inliers_vec;
    }
    return closure;
  }

  void getMatchables(const std::shared_ptr<const DescriptorManager> &q_ptr) {
    TicToc clk;
    clk.tic();
    class_matchables_query_.clear();
    class_matchables_query_.resize(buckets_.size());

    int identifier = q_ptr->getIntID();
    // ds set values
    for (int i = 0; i < q_ptr->bev_descriptors_.rows; i++) {
      auto matchable = new Tree::Matchable(
        std::make_pair(i, q_ptr->getKeyPoint(i)), q_ptr->bev_descriptors_.row(i), identifier);
      int ones_count = matchable->descriptor.count();
      for (int j = 0; j < buckets_.size(); j++) {
        if (ones_count >= buckets_[j].min_ones && ones_count <= buckets_[j].max_ones) {
          class_size_[ones_count]++;
          class_matchables_query_[j].emplace_back(matchable);
          break;
        }
      }
    }
    std::cout<<"Time getMatchables "<<clk.toc()<<std::endl;
  }

  void addToHBST(const std::shared_ptr<const DescriptorManager> &q_ptr){
    TicToc clk;
    clk.tic();
    // hbst_binary_tree_->add(matchables_query_, srrg_hbst::SplittingStrategy::SplitEven);
    DCHECK_EQ(class_matchables_query_.size(), buckets_.size());
    for (int i = 0; i < buckets_.size(); i++){
      buckets_[i].hbst_tree->add(class_matchables_query_[i], srrg_hbst::SplittingStrategy::SplitEven);
      buckets_[i].size += class_matchables_query_[i].size();
    }
    all_bevs_.emplace_back(q_ptr);
    std::cout<<"Time addToHBST "<<clk.toc()<<std::endl;
  }

  // For ease of retrieval, we first categorize by fine granularity
  void getClassMatchables(const std::shared_ptr<const DescriptorManager> &q_ptr) {
    TicToc clk;
    clk.tic();
    if(class_matchables_query_.size() == 0){
      class_matchables_query_.resize(descriptor_size_bits+1);
    }
    else{
      for (auto& vec : class_matchables_query_) {
        vec.clear(); 
      }
    }
    int identifier = q_ptr->getIntID();
    // ds set values
    for (int i = 0; i < q_ptr->bev_descriptors_.rows; i++) {
      auto matchable = new Tree::Matchable(
        std::make_pair(i, q_ptr->getKeyPoint(i)), q_ptr->bev_descriptors_.row(i), identifier);
      int ones_count = matchable->descriptor.count();
      class_matchables_query_[ones_count].emplace_back(matchable);
      class_size_[ones_count]++;
    }
    std::cout<<"Time getMatchables "<<clk.toc()<<std::endl;
  }

  void addClassToHBST(const std::shared_ptr<const DescriptorManager> &q_ptr){
    TicToc clk;
    clk.tic();
    int ones_count = 0;
    for (int b_idx = 0; b_idx < buckets_.size(); b_idx++){
      while (ones_count >= buckets_[b_idx].min_ones && ones_count <= buckets_[b_idx].max_ones && ones_count <= descriptor_size_bits) {
        if(class_matchables_query_[ones_count].size() == 0){
          ones_count++;
          continue;
        } 
        // std::cout<<"addToHBST!!! "<<std::endl;
        buckets_[b_idx].hbst_tree->add(class_matchables_query_[ones_count], srrg_hbst::SplittingStrategy::SplitEven);
        buckets_[b_idx].size += class_matchables_query_[ones_count].size();
        ones_count++;
      }
    }
    all_bevs_.emplace_back(q_ptr);
    std::cout<<"Time addToHBST "<<clk.toc()<<std::endl;
  }

};


#endif //EHPR_DESC_DB_H
