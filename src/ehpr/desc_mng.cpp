#include <glog/logging.h>
#include "ehpr/desc_mng.h"

void DescriptorManager::makeFeature(){
	TicToc clk;
	clk.tic();
	cv::Mat input_mat;
	bev_.convertTo(input_mat, CV_8U);
	// GaussianBlur(input_mat, input_mat, cv::Size(3, 3), 1.0, 1.0);

	std::vector<cv::KeyPoint> keypoints;//关键点
	cv::Mat descriptors;//描述子
	
	// extractor = cv::ORB::create();
	//-----orb!----//
	// TicToc cons_clk;
	// cons_clk.tic();
	cv::Ptr<cv::DescriptorExtractor> extractor;
	extractor = cv::ORB::create(ORB_PARAM::nfeatures, ORB_PARAM::scale_factor, ORB_PARAM::n_levels, ORB_PARAM::edge_threshold,
														ORB_PARAM::first_level, ORB_PARAM::WTA_K, cv::ORB::ScoreType(ORB_PARAM::score_type), ORB_PARAM::patch_size, ORB_PARAM::fast_threshold);
	// std::cout <<"time construct extractor"<< cons_clk.toc()<< std::endl;
	extractor->detectAndCompute(input_mat, cv::noArray(), keypoints,  descriptors);
	bev_descriptors_  = descriptors;
	//------orb----//

	// cv::Ptr<cv::SIFT> sift_detector = cv::SIFT::create(SIFT_PARAM::nfeatures, SIFT_PARAM::nOctaveLayers, SIFT_PARAM::contrastThreshold, SIFT_PARAM::edgeThreshold, SIFT_PARAM::sigma);
	// sift_detector->detect(input_mat, keypoints);
	// cv::Ptr<cv::BRISK> brisk_descriptor = cv::BRISK::create();
	// brisk_descriptor->compute(input_mat, keypoints, descriptors);
	// bev_descriptors_  = descriptors;

	//------bsift----//
	// cv::Ptr<cv::DescriptorExtractor> extractor;
	// extractor = cv::SIFT::create(SIFT_PARAM::nfeatures, SIFT_PARAM::nOctaveLayers, SIFT_PARAM::contrastThreshold, SIFT_PARAM::edgeThreshold, SIFT_PARAM::sigma);
	// extractor->detectAndCompute(input_mat, cv::noArray(), keypoints,  descriptors);
	// int numFeatures = descriptors.rows;
	// int featureLength = descriptors.cols;
	// int binaryLength = featureLength / 8; 
	// bev_descriptors_ = cv::Mat::zeros(numFeatures, binaryLength, CV_8UC1);
	// for (int i = 0; i < numFeatures; i++) {
	//     cv::Mat descriptor = descriptors.row(i);

	//     std::vector<float> vec;
	//     descriptor.copyTo(vec);
	//     std::nth_element(vec.begin(), vec.begin() + vec.size() / 2, vec.end());
	//     float median = vec[vec.size() / 2];

	//     std::bitset<BIN_KEY_DIM> bits;
	//     for (int j = 0; j < descriptor.cols; ++j) {
	//         if (descriptor.at<float>(j) >= median) {
	//             bits.set(j);
	//         }
	//     }
	//     for (int k = 0; k < binaryLength; ++k) {
	//         uint8_t byte = 0;
	//         for (int b = 0; b < 8; ++b) {
	//             if (k * 8 + b < bits.size() && bits[k * 8 + b]) {
	//                 byte |= 1 << b;
	//             }
	//         }
	//         bev_descriptors_.at<uchar>(i, k) = byte;
	//     }
	// }
	//------bsift----//
	
	int num_descriptors = bev_descriptors_.rows;
	key_height_bin_.resize(num_descriptors);
	keypoints_.resize(num_descriptors);
	// #pragma omp parallel for
	for(int index = 0; index<num_descriptors; index++){
		Eigen::Vector2d new_point(keypoints[index].pt.y, keypoints[index].pt.x);

		//---extact height_bin------//
		int row = floor(keypoints[index].pt.y);
		int col = floor(keypoints[index].pt.x);

		boost::dynamic_bitset<> key_bin(height_bin_num_);
		int begin = -1;
		int end = 1;
		for(int k = 0; k < height_bin_num_ ; k++){
			int valid_count = 0;
			int all_count = 0;
			for(int i=-cfg_.local_density_rows_; i<=cfg_.local_density_rows_; i++){
				for(int j=-cfg_.local_density_cols_; j<=cfg_.local_density_cols_; j++){
					int tmp_row = row + i;
					int tmp_col = col + j;
					if(tmp_row<0 || tmp_row>=cfg_.n_row_|| tmp_col<0 || tmp_col>=cfg_.n_col_) continue;//avoid segment fault
					all_count++;
					if(height_bin_mat_[tmp_row][tmp_col].size() == height_bin_num_ && height_bin_mat_[tmp_row][tmp_col][k]){
						valid_count++;
					}
				}
			}
			if (all_count == 0) continue;     
			if (valid_count >= all_count/3){
				key_bin.set(k, true);
			}
		}
		//record for query
		key_height_bin_[index] = key_bin;
		keypoints_[index] = new_point; //can these to be released?
	}
	std::cout<<"Time makeFeature "<<clk.toc()<<std::endl;
	std::cout << "key size " << keypoints_.size() <<" "<<key_height_bin_.size()<<std::endl;
	DCHECK_EQ(key_height_bin_.size(), keypoints_.size());
}