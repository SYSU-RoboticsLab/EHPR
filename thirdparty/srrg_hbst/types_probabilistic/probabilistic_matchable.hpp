#pragma once
#include <Eigen/Core>

#include "srrg_hbst/types/binary_matchable.hpp"

namespace srrg_hbst {

  template <typename ObjectType_,
            uint32_t descriptor_size_bits_ = 256,
            typename real_precision_       = double>
  class ProbabilisticMatchable : public BinaryMatchable<ObjectType_, descriptor_size_bits_> {
    // ds template forwarding
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef ObjectType_ ObjectType;
    typedef typename BinaryMatchable<ObjectType_, descriptor_size_bits_>::Descriptor Descriptor;
    typedef Eigen::Matrix<real_precision_, descriptor_size_bits_, 1> BitStatisticsVector;

    // ds ctor/dtor
  public:
    ProbabilisticMatchable(ObjectType object_,
                           const Descriptor& descriptor_,
                           const uint64_t& image_identifier_ = 0) :
      BinaryMatchable<ObjectType_, descriptor_size_bits_>(object_, descriptor_, image_identifier_),
      bit_probabilities(BitStatisticsVector()),
      bit_volatility(BitStatisticsVector()) {
    }

    ProbabilisticMatchable(ObjectType object_,
                           const Descriptor& descriptor_,
                           const BitStatisticsVector& bit_probabilities_,
                           const BitStatisticsVector& bit_volatility_,
                           const uint64_t& image_identifier_ = 0) :
      BinaryMatchable<ObjectType_, descriptor_size_bits_>(object_, descriptor_, image_identifier_),
      bit_probabilities(bit_probabilities_),
      bit_volatility(bit_volatility_) {
    }

    // ds wrapped constructors - only available if OpenCV is present on building system
#ifdef SRRG_HBST_HAS_OPENCV
    ProbabilisticMatchable(ObjectType object_,
                           const cv::Mat& descriptor_,
                           const uint64_t& image_identifier_ = 0) :
      BinaryMatchable<ObjectType_, descriptor_size_bits_>(object_, descriptor_, image_identifier_),
      bit_probabilities(BitStatisticsVector()),
      bit_volatility(BitStatisticsVector()) {
    }

    ProbabilisticMatchable(ObjectType object_,
                           const cv::Mat& descriptor_,
                           const BitStatisticsVector& bit_probabilities_,
                           const BitStatisticsVector& bit_volatility_,
                           const uint64_t& image_identifier_ = 0) :
      BinaryMatchable<ObjectType_, descriptor_size_bits_>(object_, descriptor_, image_identifier_),
      bit_probabilities(bit_probabilities_),
      bit_volatility(bit_volatility_) {
    }
#endif

    ~ProbabilisticMatchable() {
    }

    // ds attributes
  public:
    // ds statistical data: bit probabilities
    BitStatisticsVector bit_probabilities;

    // ds statistical data: bit volatity info, maximum number of appearances where the bit was
    // stable - currently not used
    BitStatisticsVector bit_volatility;
  };

} // namespace srrg_hbst
