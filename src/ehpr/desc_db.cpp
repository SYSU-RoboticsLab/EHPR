#include "ehpr/desc_db.h"
#include <glog/logging.h>
void drawMatchesAndSave(const std::vector<map_closures::PointPair> & correspondences,
						const std::vector<int> & inliers,
						const cv::Mat& query_image, const cv::Mat& ref_image,
						const std::string& output_path) {	
	cv::Mat img1_color, img2_color;
	double scale_factor = 4;
	cv::cvtColor(query_image, img1_color, cv::COLOR_GRAY2BGR);
	cv::cvtColor(ref_image, img2_color, cv::COLOR_GRAY2BGR);

	cv::resize(img1_color, img1_color, cv::Size(), scale_factor, scale_factor);
	cv::resize(img2_color, img2_color, cv::Size(), scale_factor, scale_factor);

  int blank_width = 20;
  cv::Mat output_image(img1_color.rows, img1_color.cols + img2_color.cols + blank_width, CV_8UC3, cv::Scalar(255, 255, 255));
  
  //set blank space color
  output_image(cv::Rect(img1_color.cols, 0, blank_width, img1_color.rows)).setTo(cv::Scalar(255, 255,255));

  img1_color.copyTo(output_image(cv::Rect(0, 0, img1_color.cols, img1_color.rows)));
  img2_color.copyTo(output_image(cv::Rect(img1_color.cols + blank_width, 0, img2_color.cols, img2_color.rows)));

  int count = 0;
  for (int inlier_idx : inliers) {
    cv::Point2d query_pt(correspondences[inlier_idx].query(1)*scale_factor, correspondences[inlier_idx].query(0)*scale_factor);
    cv::Point2d ref_pt(correspondences[inlier_idx].ref(1)*scale_factor + img1_color.cols+blank_width, correspondences[inlier_idx].ref(0)*scale_factor);
    
    // Draw at the points
    cv::circle(output_image, ref_pt, 5, cv::Scalar(255, 0, 255), 2);
    cv::circle(output_image, query_pt, 5, cv::Scalar(255, 0, 255), 2);  
    cv::line(output_image, ref_pt, query_pt, cv::Scalar(71, 173, 112), 2);

    // Add text with the feature point index
    // cv::putText(output_image, std::to_string(count), query_pt-cv::Point2d(10,0), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 50, 50), 1);
    // cv::putText(output_image, std::to_string(count), ref_pt+cv::Point2d(10,0), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 50, 50), 1);
    count++;
  }
	// cv::resize(output_image, output_image, cv::Size(), 1/scale_factor, 1/scale_factor);
    // Save the output image
    cv::imwrite(output_path, output_image);
}

double DescriptorDB::calcauInlierRatio(int qr_id){
	double inliers_distance_threshold = 3.0;
	int inlier_cnt = 0;
	int match_count = 0;
	for (const auto& id_match : descriptor_matches_) {
			int ref_id = id_match.first;
			std::for_each(id_match.second.cbegin(), id_match.second.cend(),
			[&](const Tree::Match&match) {
						Eigen::Isometry3d gt_relative;
						if(db_imu_vec.empty()&&qr_imu_vec.empty()){
							gt_relative = db_gt_vec[ref_id].inverse()*qr_gt_vec[qr_id];
						}
						else{
							gt_relative = db_imu_vec[ref_id]*
														db_gt_vec[ref_id].inverse()*qr_gt_vec[qr_id]
														*qr_imu_vec[qr_id].inverse();;
						}
						Eigen::Vector3d translation_3d = gt_relative.translation();
						double x = translation_3d.x();
						double y = translation_3d.y();
						Eigen::Matrix3d rotation_3d = gt_relative.rotation();
						double yaw = std::atan2(rotation_3d(1, 0), rotation_3d(0, 0));
						Eigen::Isometry2d gt_relative_2d = Eigen::Isometry2d::Identity();
						gt_relative_2d.translate(Eigen::Vector2d(x, y)); 
						gt_relative_2d.rotate(yaw); 

						Eigen::Vector2d ref_point = Eigen::Vector2d((match.object_references[0].second.x()-cfg_.n_row_/2.0)*cfg_.reso_row_, 
																												(match.object_references[0].second.y()-cfg_.n_col_/2.0)*cfg_.reso_col_);
						Eigen::Vector2d query_point = Eigen::Vector2d((match.object_query.second.x()-cfg_.n_row_/2.0)*cfg_.reso_row_, 
																													(match.object_query.second.y()-cfg_.n_col_/2.0)*cfg_.reso_col_);       
						if ((gt_relative_2d * query_point - ref_point).norm() < inliers_distance_threshold)
							inlier_cnt++;
						match_count++;
				});
	}
	double ratio;
	if(match_count>0){
		ratio = 1.0*inlier_cnt/match_count;
	}
	else{
		ratio = 0.0;
	}
	return ratio;
}

int DescriptorDB::queryFeatureHamming(const std::shared_ptr<const DescriptorManager> &q_ptr,
                    std::vector<int> &cand_indx,
                    std::vector<double> &cand_corr,
                    Eigen::Matrix4d &pose) {
	int distup = cfg_.dist_thres_;
	cand_correspond_.clear();
	descriptor_matches_.clear();

	TicToc clk;
	clk.tic();
	stp.start();
	for(int i = 0; i < class_matchables_query_.size(); i++){
		// std::cout<<"class_matchables_query_"<<i<<" "<<class_matchables_query_[i].size()<<std::endl;
		// int search_range_min = buckets_[i].min_ones - distup;
		// int search_range_max = buckets_[i].max_ones + distup;
		if(class_matchables_query_[i].empty()) continue; 
		int ones_count = i;
		int search_range_min = ones_count- distup;
		int search_range_max = ones_count + distup;
		for (int j = 0; j < buckets_.size(); j++){       
		if((buckets_[j].min_ones <= search_range_min && buckets_[j].max_ones >= search_range_min)
				||(buckets_[j].min_ones >= search_range_min && buckets_[j].max_ones <= search_range_max)
				||(buckets_[j].min_ones <= search_range_max && buckets_[j].max_ones >= search_range_max)){
				Tree::MatchVectorMap tmp_matches;
				buckets_[j].hbst_tree->match(class_matchables_query_[i], tmp_matches, distup);

				for (const auto& entry : tmp_matches) {
					int key = entry.first;
					const auto& matches = entry.second;
					if (descriptor_matches_.find(key) != descriptor_matches_.end()) {
							descriptor_matches_[key].insert(descriptor_matches_[key].end(), matches.begin(), matches.end());
					} else {
						descriptor_matches_[key] = matches;
					}
				}
			}
		}
	}
	// hbst_binary_tree_->match(matchables_query_, descriptor_matches_,
	//                                100);
	stp.record("feature vote");
	std::cout<<"Time feature vote "<<clk.toc()<<std::endl;

	// human read to do delete
	int match_count = 0;
	int match_size = 0 ; 
	for (const auto& match : descriptor_matches_) {
			match_count += int(match.second.size());
			match_size++;
			// std::cout<<match.first<<" "<< match.second.size()<<std::endl; 
	}
	std::cout<<"match_size"<<match_size<<" match_count"<<match_count<<std::endl;

	// std::cout<<std::endl;
	std::vector<int> indices(descriptor_matches_.size()); 
	std::iota(indices.begin(), indices.end(), 0);//增量赋值0 1 2 3 4

	auto compare_closure_candidates = [](ClosureCandidate a,
																					const ClosureCandidate &b) -> ClosureCandidate {
			return a.number_of_inliers > b.number_of_inliers ? a : b;
	};
	using iterator_type = std::vector<int>::const_iterator;

	clk.tic();
	stp.start();
	const ClosureCandidate &bestclosure = tbb::parallel_reduce(
			tbb::blocked_range<iterator_type>{indices.cbegin(), indices.cend()}, ClosureCandidate(),
			[&](const tbb::blocked_range<iterator_type> &r,
					ClosureCandidate candidate) -> ClosureCandidate {
					return std::transform_reduce(
							r.begin(), r.end(), candidate, compare_closure_candidates, [&](const auto &ref_id) {
									const bool is_far_enough = std::abs(static_cast<int>(ref_id) - q_ptr->getIntID()) > cfg_.skip_num_;
									// const bool is_far_enough = true;
									return is_far_enough ? ValidateClosure(ref_id, q_ptr->getIntID()) : ClosureCandidate();
							});
			},
			compare_closure_candidates);
	stp.record("findBestMatch");
	std::cout<<"Time findBestMatch "<<clk.toc()<<std::endl;

	if(bestclosure.number_of_inliers >= 5){ //to be tune for pr evaluation
			const std::vector<int>& inliers_vec = bestclosure.inlier_vec;
			int best_id = bestclosure.source_id;
			//cand_ptrs.emplace_back(all_bevs_[best_id]);
			cand_indx.emplace_back(all_bevs_[best_id]->attach_idx_);//data seq idx for evaluate with gt
			cand_corr.emplace_back(bestclosure.number_of_inliers);
			pose = bestclosure.pose;
			std::cout <<"bev_" << best_id <<" inlier size " <<bestclosure.number_of_inliers<<std::endl;
			
			double x  = pose(0, 3);
			double y  = pose(1, 3);
			double extracted_yaw = std::atan2(pose(1, 0), pose(0, 0));
			std::cout << "Extracted x: y: yaw (degrees): "<<clk.toc()<<" "<<x<<" "<<y<<" "<< extracted_yaw * 180.0 / M_PI << std::endl;
			
			const Tree::MatchVector &matches = descriptor_matches_.at(bestclosure.source_id);
			cand_correspond_[best_id].resize(matches.size());
			std::transform(matches.cbegin(), matches.cend(), cand_correspond_[best_id].begin(),
				[&](const Tree::Match &match) {
					Eigen::Vector2d ref_point = match.object_references[0].second;
					Eigen::Vector2d query_point = match.object_query.second;
					int ref_idx = match.object_references[0].first;
					int qur_idx = match.object_query.first;
					return map_closures::PointPair(ref_point, query_point, ref_idx, qur_idx);
			});
			#if SAVE_MID_FILE
			std::string f_name =
							std::string(PJSRCDIR) + "/results/match_comp_img/lcor_" + q_ptr->getStrID() + "-" + all_bevs_[best_id]->getStrID() +
							".png";
			drawMatchesAndSave(cand_correspond_[best_id], inliers_vec,
							   q_ptr->getBevImage(), all_bevs_[best_id]->getBevImage(), f_name);
			#endif

			//------register h use inliers---------//
			clk.tic();
			stp.start();
			const auto &[qur_hmax, qur_hmin] = q_ptr->getHightMinMax();
			const auto &[ref_hmax, ref_hmin] = all_bevs_[best_id]->getHightMinMax();
			int ref_bin_num, qur_bin_num;
			float ref_bin_res, qur_bin_res;

			q_ptr->getBinInfo(qur_bin_num, qur_bin_res);
			all_bevs_[best_id]->getBinInfo(ref_bin_num, ref_bin_res);
			std::vector<int> score_sub(ref_bin_num+qur_bin_num, 0);
			std::vector<int> score_add(ref_bin_num+qur_bin_num, 0);
			std::vector<int> score(ref_bin_num+qur_bin_num, 0);

			std::cout<<"qur_hmax/min "<<qur_hmax<<" "<<qur_hmin<<std::endl;
			std::cout<<"ref_hmax/min "<<ref_hmax<<" "<<ref_hmin<<std::endl;
			
			int count = 0;
			for(int inlier_idx : inliers_vec){
				// std::cout<<" getbinary "<<std::endl;
				boost::dynamic_bitset<> qur_bin = q_ptr->getKeyHeightBin(cand_correspond_[best_id][inlier_idx].qur_idx);
				boost::dynamic_bitset<> ref_bin = all_bevs_[best_id]->getKeyHeightBin(cand_correspond_[best_id][inlier_idx].ref_idx);
				
				int qur_ori =  floor((0.0 - qur_hmin)/qur_bin_res);
				int ref_ori = floor((0.0 - ref_hmin)/ref_bin_res);
				int qur_begin = qur_bin.find_first();
				int ref_begin = ref_bin.find_first();
				int qur_end = std::min(qur_bin_num-1, std::max(qur_ori, find_last(qur_bin)));
				int ref_end = std::min(ref_bin_num-1, std::max(ref_ori, find_last(ref_bin))); 
				//
				//bin display
				// std::cout<<count<<std::endl;
				// std::cout<<" qursize "<<qur_bin.size()<<" "<<qur_begin<<" "<<qur_end<<std::endl;
				// std::cout<<" refsize "<<ref_bin.size()<<" "<<ref_begin<<" "<<ref_end<<std::endl;
				// std::cout<<" qur "<<qur_bin<<std::endl;
				// std::cout<<" ref "<<ref_bin<<std::endl;

				count++;
				if(qur_begin == -1 || ref_begin == -1){
						std::cout<<"no 1"<<std::endl;
						continue;
				}

						//offset = 0 means ref_bin_end align qur_bin_begin
				for(int offset = 0; offset < score_add.size(); offset++){ 
					//ref_bin = ref_bin<<1;
					int ref_end_qur = offset -(ref_bin.size() - ref_end)+1;//only in lower bound of qur(air) and upper bound of ref(ground)
					int ref_begin_qur = offset -(ref_bin.size() - ref_begin)+1;
					int com_end_qur = std::min(qur_end, ref_end_qur);
					int com_begin_qur = std::max(qur_begin, ref_begin_qur);
					if(com_end_qur >= com_begin_qur){  //&& offset <= qur_end + ref_bin.size()-ref_end-1??{
						for(int index = com_begin_qur; index <= com_end_qur; index++){
							if (qur_bin[index] == ref_bin[index-offset+ref_bin.size()-1]){
							score_add[offset]++;
							}
							else{
							score_sub[offset]--;
							}
						}
					}
				}
			}

			tbb::parallel_for(tbb::blocked_range<size_t>(0, score_add.size()),
									[&](const tbb::blocked_range<size_t>& r) {
											for (size_t i = r.begin(); i < r.end(); ++i) {
													score[i] = score_add[i] + score_sub[i];
											}
									});
			
			std::vector<int>::iterator itMax = max_element(score.begin(), score.end());
			int op_score = *itMax;
			// int optimal_offset = itMax - score.begin();
			int optimal_offset = 0;
			int op_score_add = std::numeric_limits<int>::min();

			for (size_t i = 0; i < score.size(); ++i) {
			if (score[i] == op_score && score_add[i] > op_score_add){
					op_score_add = score_add[i];
					optimal_offset = i;
			}
			}
			
			if(op_score_add == 0){
				std::cout<<"No optimal height!"<<std::endl;
			}
			else{
				double relative_height = (qur_hmin + (optimal_offset+0.5) * qur_bin_res - (ref_hmax));//top bin is aligned
				std::cout<<optimal_offset<<"offset with score "<<op_score<<" optimal height is "<<relative_height<<std::endl;
				pose(2,3) = relative_height;   //ref axe in qur axe i.e. ref point to qur 
			}
			stp.record("height align");
			align_time = clk.toc();
			std::cout<<"Time height codealign "<<align_time<<std::endl;
			
	}
	return match_count;
}