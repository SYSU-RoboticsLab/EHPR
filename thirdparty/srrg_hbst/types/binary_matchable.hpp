#pragma once
#include <assert.h>
#include <bitset>
#include <map>
#include <stdint.h>
#include <vector>

// ds if opencv is present on building system
#ifdef SRRG_HBST_HAS_OPENCV
#include <opencv2/core/version.hpp>
#include <opencv2/opencv.hpp>
#endif

namespace srrg_hbst {

  //! @class default matching object (wraps the input descriptors and more)
  //! @param descriptor_size_bits_ number of bits for the native descriptor
  template <typename ObjectType_, uint32_t descriptor_size_bits_ = 256>
  class BinaryMatchable {
    // ds exports
  public:
    //! @brief descriptor type (extended by augmented bits, no effect if zero)
    using Descriptor = std::bitset<descriptor_size_bits_>;
    using ObjectType = ObjectType_;
    using ObjectMap  = std::map<uint64_t, ObjectType>;

    // ds shared properties
  public:
    //! @brief descriptor size in bits (for all matchables)
    static constexpr uint32_t descriptor_size_bits = descriptor_size_bits_;

    //! @brief descriptor size in bytes (for all matchables, size after the number of augmented
    //! bits)
    static constexpr uint32_t raw_descriptor_size_bytes = descriptor_size_bits_ / 8;

    //! @brief descriptor size, bits in whole bytes (corresponds to descriptor_size_bits for a
    //! byte-wise descriptor)
    static constexpr uint32_t descriptor_size_bits_in_bytes = raw_descriptor_size_bytes * 8;

    //! @brief overflowing single bits (normally 0)
    static constexpr uint32_t descriptor_size_bits_overflow =
      descriptor_size_bits - descriptor_size_bits_in_bytes;

    // ds ctor/dtor
  public:
    //! @brief default constructor: DISABLED
    BinaryMatchable() = delete;

    //! @brief constructor with an object pointer for association
    //! @param[in] object_ associated object (note that HBST takes ownership of this object and
    //! invalidates the variable)
    //! @param[in] descriptor_ HBST descriptor
    //! @param[in] image_identifier_ reference to image on which the descriptors have been computed
    //! (optional)
    BinaryMatchable(ObjectType object_,
                    const Descriptor& descriptor_,
                    const uint64_t& image_identifier_ = 0) :
      descriptor(descriptor_),
      number_of_objects(1),
      _image_identifier(image_identifier_),
      _object(std::move(object_)) {
      objects.insert(std::make_pair(_image_identifier, _object));
      assert(number_of_objects == objects.size());
    }

    //! @brief constructor from object map
    BinaryMatchable(ObjectMap objects_, const Descriptor& descriptor_) :
      descriptor(descriptor_),
      objects(objects_),
      number_of_objects(objects_.size()),
      _image_identifier(objects_.begin()->first),
      _object(objects_.begin()->second) {
      assert(number_of_objects == objects.size());
    }

// ds wrapped constructors - only available if OpenCV is present on building system
// ds this choice is slower since we have to perform a conversion: getDescriptor()
#ifdef SRRG_HBST_HAS_OPENCV

    //! @brief constructor with an object pointer for association
    //! @param[in] object_ associated object
    //! @param[in] descriptor_ OpenCV descriptor
    //! @param[in] image_identifier_ reference to image on which the descriptors have been computed
    //! (optional)
    BinaryMatchable(ObjectType object_,
                    const cv::Mat& descriptor_,
                    const uint64_t& image_identifier_ = 0) :
      BinaryMatchable(object_, getDescriptor(descriptor_), image_identifier_) {
    }
#endif

    //! @brief default destructor
    virtual ~BinaryMatchable() {
      objects.clear();
    }

    // ds functionality
  public:
    //! @brief computes the classic Hamming descriptor distance between this and another matchable
    //! @param[in] matchable_query_ the matchable to compare this against
    //! @returns the matching distance as integer
    inline const uint32_t
    distance(const BinaryMatchable<ObjectType_, descriptor_size_bits_>* matchable_query_) const {
      return (matchable_query_->descriptor ^ this->descriptor).count();
    }

#ifdef SRRG_MERGE_DESCRIPTORS
    //! @brief merges a matchable with THIS matchable (desirable when having to store identical
    //! descriptors)
    //! @param[in] matchable_ the matchable to merge with THIS
    inline void merge(const BinaryMatchable<ObjectType_, descriptor_size_bits_>* matchable_) {
      objects.insert(matchable_->objects.begin(), matchable_->objects.end());
      number_of_objects += matchable_->objects.size();
      assert(number_of_objects == objects.size());
    }

    //! @brief merges a matchable with THIS matchable (desirable when having to store identical
    //! descriptors)
    //! @brief this method has been created for quick matchable merging where matchable_ only
    //! contains a single entry for identifier and pointer
    //! @param[in] matchable_ the matchable to merge with THIS
    inline void mergeSingle(const BinaryMatchable<ObjectType_, descriptor_size_bits_>* matchable_) {
      objects.insert(std::make_pair(matchable_->_image_identifier, std::move(matchable_->_object)));
      ++number_of_objects;
      assert(number_of_objects == objects.size());
    }
#endif

    //! @brief enables manual update of the inner linked object
    inline void setObject(ObjectType object_) {
      _object = std::move(object_);
    }

    //! @brief enables manual update of all linked objects (without changing the referenced image
    //! number)
    inline void setObjects(ObjectType object_) {
      setObject(object_);
      for (auto& object : objects) {
        object.second = std::move(object_);
      }
    }

#ifdef SRRG_HBST_HAS_OPENCV
    //! @brief descriptor wrapping - only available if OpenCV is present on building system
    //! @param[in] descriptor_cv_ opencv descriptor to convert into HBST format
    static inline Descriptor getDescriptor(const cv::Mat& descriptor_cv_) {
      // ds buffer
      Descriptor binary_descriptor(descriptor_size_bits_);

      // ds set original descriptor string after augmentation
      for (uint64_t byte_index = 0; byte_index < raw_descriptor_size_bytes; ++byte_index) {
        const uint32_t bit_index_start = byte_index * 8;

        // ds grab a byte and convert it to a bitset so we can access the single bits
        const std::bitset<8> descriptor_byte(descriptor_cv_.at<uchar>(byte_index));

        // ds set bitstring
        for (uint8_t v = 0; v < 8; ++v) {
          binary_descriptor[bit_index_start + v] = descriptor_byte[v];
        }
      }

      // ds check if we have extra bits (less than 1 byte i.e. <= 7 bits)
      if (descriptor_size_bits_overflow > 0) {
        // ds get last byte (not fully set)
        const std::bitset<8> descriptor_byte(descriptor_cv_.at<uchar>(raw_descriptor_size_bytes));

        // ds only set the remaining bits
        for (uint32_t v = 0; v < descriptor_size_bits_overflow; ++v) {
          binary_descriptor[descriptor_size_bits_in_bytes + v] =
            descriptor_byte[8 - descriptor_size_bits_overflow + v];
        }
      }
      return binary_descriptor;
    }
#endif

    // ds attributes
  public:
    //! @brief descriptor data string vector
    const Descriptor descriptor;

    //! @brief a connected object correspondences - when using this field one must ensure the
    //! permanence of the referenced object!
    ObjectMap objects;

    //! @brief quick access to the number of contained objects/image_identifiers (default: 1)
    uint64_t number_of_objects;

    // ds fast access (for a matchable with only single values, internal only)
  protected:
    //! @brief single value access only: linked object to group of descriptors (e.g. an image or
    //! image index)
    const uint64_t _image_identifier;

    //! @brief single value access only: linked object to descriptor (e.g. keypoint or an index)
    ObjectType _object;

    //! @brief allow direct access for processing classes
    template <typename BinaryNodeType_>
    friend class BinaryTree;
  };

  // ds come on c++11
  template <typename ObjectType_, uint32_t descriptor_size_bits_>
  constexpr uint32_t BinaryMatchable<ObjectType_, descriptor_size_bits_>::descriptor_size_bits;
  template <typename ObjectType_, uint32_t descriptor_size_bits_>
  constexpr uint32_t BinaryMatchable<ObjectType_, descriptor_size_bits_>::raw_descriptor_size_bytes;
  template <typename ObjectType_, uint32_t descriptor_size_bits_>
  constexpr uint32_t
    BinaryMatchable<ObjectType_, descriptor_size_bits_>::descriptor_size_bits_in_bytes;
  template <typename ObjectType_, uint32_t descriptor_size_bits_>
  constexpr uint32_t
    BinaryMatchable<ObjectType_, descriptor_size_bits_>::descriptor_size_bits_overflow;

  template <typename ObjectType_>
  using BinaryMatchable128 = BinaryMatchable<ObjectType_, 128>;
  template <typename ObjectType_>
  using BinaryMatchable256 = BinaryMatchable<ObjectType_, 256>;
  template <typename ObjectType_>
  using BinaryMatchable512 = BinaryMatchable<ObjectType_, 512>;

} // namespace srrg_hbst
