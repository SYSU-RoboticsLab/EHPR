#ifndef _VOXELMAP_HPP_
#define _VOXELMAP_HPP_

#include <unordered_map>
#include <stdio.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <Eigen/Core>

#define HASH_P 116101
#define MAX_N 10000000000
#define MAX_FRAME_N 20000

class VOXEL_LOC {
public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
      : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOC &other) const {
    return (x == other.x && y == other.y && z == other.z);
  }
};


// Hash value
template <> struct std::hash<VOXEL_LOC> {
  int64_t operator()(const VOXEL_LOC &s) const {
    using std::hash;
    using std::size_t;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};

struct M_POINT {
  float xyz[3];
  float attach_info;
  int count = 0;
};


void voxelDownSample(pcl::PointCloud<pcl::PointXYZ> &pl_feat,
                         double voxel_size) {
  int intensity = rand() % 255;
  if (voxel_size < 0.01) {
    return;
  }
  std::unordered_map<VOXEL_LOC, M_POINT> voxel_map;
  uint plsize = pl_feat.size();

  for (uint i = 0; i < plsize; i++) {
    pcl::PointXYZ &p_c = pl_feat[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      iter->second.xyz[0] += p_c.x;
      iter->second.xyz[1] += p_c.y;
      iter->second.xyz[2] += p_c.z;
      iter->second.count++;
    } else {
      M_POINT anp;
      anp.xyz[0] = p_c.x;
      anp.xyz[1] = p_c.y;
      anp.xyz[2] = p_c.z;
      anp.count = 1;
      voxel_map[position] = anp;
    }
  }
  plsize = voxel_map.size();
  pl_feat.clear();
  pl_feat.resize(plsize);

  uint i = 0;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter) {
    pl_feat[i].x = iter->second.xyz[0] / iter->second.count;
    pl_feat[i].y = iter->second.xyz[1] / iter->second.count;
    pl_feat[i].z = iter->second.xyz[2] / iter->second.count;
    i++;
  }
}

class OctoTree {
public:
//   ConfigSetting config_setting_;
  std::vector<Eigen::Vector3d> voxel_points_;
  int layer_;
  int octo_state_; // 0 is end of tree, 1 is not
  int merge_num_ = 0;
  bool is_project_ = false;
//   std::vector<Eigen::Vector3d> proj_normal_vec_;
//   check 6 direction: x,y,z,-x,-y,-z
//   bool is_check_connect_[6];
//   bool connect_[6];
//   OctoTree *connect_tree_[6];

  bool is_publish_ = false;
  Eigen::Vector3d voxel_center_; // x, y, z
  // float quater_length_;
  // bool init_octo_;
  OctoTree() {  //(const ConfigSetting &config_setting)  : config_setting_(config_setting)
    voxel_points_.clear();
    octo_state_ = 0;
    layer_ = 0;
    // for (int i = 0; i < 6; i++) {
    //   is_check_connect_[i] = false;
    //   connect_[i] = false;
    //   connect_tree_[i] = nullptr;
    // }
  }
//   void init_octo_tree();
};

struct VoxelMapConfig {
  double voxel_size = 1.0;
  double max_distance = 5;
  double remove_distance = 80;
  int max_points_per_voxel = 10;
};

class VoxelMap {
    const VoxelMapConfig cfg_;
    std::unordered_map<VOXEL_LOC, OctoTree*> voxel_map_;
public:
  Eigen::Vector3d xyz_max_, xyz_min_;
  explicit VoxelMap(const VoxelMapConfig &config):cfg_(config){
    xyz_max_ = -10000.0 * Eigen::Vector3d::Ones();
    xyz_min_ = 10000.0 * Eigen::Vector3d::Ones();
  }
	inline void Clear() {
    for (auto& pair : voxel_map_) {
        delete pair.second; 
    }
    voxel_map_.clear(); 
  }
	inline bool Empty() const { return voxel_map_.empty(); }
	void getAllPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
			cloud->clear();
      // cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
			for (const auto& kv : voxel_map_) {
				const auto& octo_tree = kv.second;
				for (const auto& point : octo_tree->voxel_points_) {
						pcl::PointXYZ pcl_point;
						pcl_point.x = point.x();
						pcl_point.y = point.y();
						pcl_point.z = point.z();
						cloud->points.push_back(pcl_point);
				}
			}
			cloud->width = cloud->points.size();
			cloud->height = 1;  // 无序点云
			cloud->is_dense = true;  // 假设没有 NaN 点
	}

	void addPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud) {
		const double map_resolution = std::sqrt(cfg_.voxel_size * cfg_.voxel_size / cfg_.max_points_per_voxel);
		for (size_t i = 0; i < input_cloud->size(); i++) {
			Eigen::Vector3d p_c(input_cloud->points[i].x, input_cloud->points[i].y, input_cloud->points[i].z);
			double loc_xyz[3];
			for (int j = 0; j < 3; j++) {
        // xyz_max_[j] = std::max(xyz_max_[j], p_c[j]);
        // xyz_min_[j] = std::min(xyz_min_[j], p_c[j]);

				loc_xyz[j] = p_c[j] / cfg_.voxel_size;
				if (loc_xyz[j] < 0) {
					loc_xyz[j] -= 1.0;
				}
			}
			VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
			auto iter = voxel_map_.find(position);
			if (iter != voxel_map_.end()) {
				auto & voxel_points = iter->second->voxel_points_;
				if (voxel_points.size() == cfg_.max_points_per_voxel ||
					std::any_of(voxel_points.cbegin(), voxel_points.cend(),
											[&](const auto &voxel_point) {
													return (voxel_point - p_c).norm() < map_resolution;
											})) {
									continue;
				}
				voxel_points.push_back(p_c);
			} else {
				OctoTree *octo_tree = new OctoTree();
				voxel_map_[position] = octo_tree;
				voxel_map_[position]->voxel_points_.push_back(p_c);
				voxel_map_[position]->voxel_center_ = Eigen::Vector3d(loc_xyz[0] * cfg_.voxel_size + cfg_.voxel_size/2,
																															loc_xyz[1] * cfg_.voxel_size + cfg_.voxel_size/2,
																															loc_xyz[2] * cfg_.voxel_size + cfg_.voxel_size/2);
			}
		}
	}

	void removePointsFarFromLocation(const Eigen::Vector3d &origin) {
		double max_distance2 = cfg_.remove_distance * cfg_.remove_distance;
		for (auto it = voxel_map_.begin(); it != voxel_map_.end();) {
			Eigen::Vector3d &pt = it->second->voxel_center_;
			if ((pt - origin).squaredNorm() >= (max_distance2)) {
          delete it->second;
					it = voxel_map_.erase(it);
			}else {
					++it;
			}
		}
	}

  void adaptiveDownsample(int sample_num,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr &pl_out) {
    pl_out->points.clear();
    pl_out->points.reserve(voxel_map_.size() * sample_num);
    for (auto voxel: voxel_map_) {
      if (sample_num == 1) {
          pcl::PointXYZ pcl_point;
          pcl_point.x = voxel.second->voxel_points_[0].x();
          pcl_point.y = voxel.second->voxel_points_[0].y();
          pcl_point.z = voxel.second->voxel_points_[0].z();
          pl_out->points.push_back(pcl_point);
      } else {
        int pt_size = voxel.second->voxel_points_.size();
        if (pt_size > sample_num) {
          int step = pt_size / sample_num;
          for (int i = 0; i < sample_num; ++i) {
            pcl::PointXYZ pcl_point;
            pcl_point.x = voxel.second->voxel_points_[i * step].x();
            pcl_point.y = voxel.second->voxel_points_[i * step].y();
            pcl_point.z = voxel.second->voxel_points_[i * step].z();
            pl_out->points.push_back(pcl_point);
          }
        } else {
          for (int i = 0; i < pt_size; ++i) {
            pcl::PointXYZ pcl_point;
            pcl_point.x = voxel.second->voxel_points_[i].x();
            pcl_point.y = voxel.second->voxel_points_[i].y();
            pcl_point.z = voxel.second->voxel_points_[i].z();
            pl_out->points.push_back(pcl_point);
          }
        }
      }
    }
  }
};


#endif