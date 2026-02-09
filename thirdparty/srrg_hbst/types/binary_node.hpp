#pragma once
#include <cmath>
#include <random>

#include "binary_match.hpp"

namespace srrg_hbst {

  //! @brief leaf spawning modes
  enum SplittingStrategy { DoNothing, SplitEven, SplitUneven, SplitRandomUniform };

  template <typename BinaryMatchableType_, typename real_type_ = double>
  class BinaryNode {
    // ds readability
    using Node = BinaryNode<BinaryMatchableType_, real_type_>;

    // ds exports
  public:
    using BaseNode        = Node;
    using Matchable       = BinaryMatchableType_;
    using MatchableVector = std::vector<Matchable*>;
    using Descriptor      = typename Matchable::Descriptor;
    using real_type       = real_type_;
    using Match           = BinaryMatch<Matchable, real_type>;

    //! @brief header for de/serialization TODO fuse with attributes
    struct Header {
      Header(const uint64_t& depth_) : depth(depth_) {
      }
      Header() : Header(0) {
      }
      uint64_t depth;
      uint64_t number_of_matchables_uncompressed = 0;
      uint64_t number_of_matchables_compressed   = 0;
    };

    // ds ctor/dtor
  public:
    // ds access only through this constructor: no mask provided
    BinaryNode(const MatchableVector& matchables_,
               const SplittingStrategy& train_mode_ = SplittingStrategy::SplitEven) :
      Node(nullptr, 0, matchables_, Descriptor().set(), train_mode_) {
    }

    // ds access only through this constructor: mask provided
    BinaryNode(const MatchableVector& matchables_,
               Descriptor bit_mask_,
               const SplittingStrategy& train_mode_ = SplittingStrategy::SplitEven) :
      Node(nullptr, 0, matchables_, bit_mask_, train_mode_) {
    }

    // ds the default constructor is triggered by subclasses - the responsibility of attribute
    // initialization is left to the subclass ds this is required, since we do not want to trigger
    // the automatic leaf spawning of the baseclass in a subclass
    BinaryNode() {
    }

    // ds destructor: recursive destruction of child nodes (risky but readable)
    virtual ~BinaryNode() {
      delete left;
      delete right;
    }

    // ds access
  public:
    // ds create leafs (external use intented)
    virtual const bool spawnLeafs(const SplittingStrategy& train_mode_) {
      assert(!has_leafs);
      _header.number_of_matchables_compressed = matchables.size();

      // ds exit if maximum depth is reached
      if (_header.depth == maximum_depth) {
        return false;
      }

      // ds exit if we have insufficient data
      if (_header.number_of_matchables_uncompressed < maximum_leaf_size) {
        return false;
      }

      // ds affirm initial situation
      index_split_bit         = -1;
      number_of_on_bits_total = 0;
      partitioning            = maximum_partitioning;

      // ds for balanced splitting
      switch (train_mode_) {
        case SplittingStrategy::SplitEven: {
          // ds we have to find the split for this node - scan all indices
          for (uint32_t bit_index = 0; bit_index < Matchable::descriptor_size_bits; ++bit_index) {
            // ds if this index is available in the mask
            if (bit_mask[bit_index]) {
              // ds temporary set bit count
              uint64_t number_of_set_bits = 0;

              // ds compute distance for this index (0.0 is perfect)
              const double partitioning_current =
                std::fabs(0.5 - _getSetBitFraction(bit_index, matchables, number_of_set_bits));

              // ds if better
              if (partitioning_current < partitioning) {
                partitioning            = partitioning_current;
                number_of_on_bits_total = number_of_set_bits;
                index_split_bit         = bit_index;

                // ds finalize loop if maximum target is reached
                if (partitioning == 0)
                  break;
              }
            }
          }
          break;
        }
        case SplittingStrategy::SplitUneven: {
          partitioning = 0;

          // ds we have to find the split for this node - scan all indices
          for (uint32_t bit_index = 0; bit_index < Matchable::descriptor_size_bits; ++bit_index) {
            // ds if this index is available in the mask
            if (bit_mask[bit_index]) {
              // ds temporary set bit count
              uint64_t number_of_set_bits = 0;

              // ds compute distance for this index (0.0 is perfect)
              const double partitioning_current =
                std::fabs(0.5 - _getSetBitFraction(bit_index, matchables, number_of_set_bits));

              // ds if worse
              if (partitioning_current > partitioning) {
                partitioning            = partitioning_current;
                number_of_on_bits_total = number_of_set_bits;
                index_split_bit         = bit_index;

                // ds finalize loop if maximum target is reached
                if (partitioning == 0.5)
                  break;
              }
            }
          }
          break;
        }
        case SplittingStrategy::SplitRandomUniform: {
          // ds compute available bits
          std::vector<uint32_t> available_bits;
          for (uint32_t bit_index = 0; bit_index < Matchable::descriptor_size_bits; ++bit_index) {
            // ds if this index is available in the mask
            if (bit_mask[bit_index]) {
              available_bits.push_back(bit_index);
            }
          }

          // ds if bits are available
          if (available_bits.size() > 0) {
            std::uniform_int_distribution<uint32_t> available_indices(0, available_bits.size() - 1);

            // ds sample uniformly at random
            index_split_bit = available_bits[available_indices(Node::random_number_generator)];

            // ds compute distance for this index (0.0 is perfect)
            partitioning = std::fabs(
              0.5 - _getSetBitFraction(index_split_bit, matchables, number_of_on_bits_total));
          }
          break;
        }
        default: { throw std::runtime_error("invalid leaf spawning mode"); }
      }

      // ds if best was found and the partitioning is sufficient (0 to 0.5) - we can spawn leaves
      if (index_split_bit != -1 && partitioning < maximum_partitioning) {
        // ds get a mask copy
        Descriptor bit_mask_previous(bit_mask);

        // ds update mask for leafs
        bit_mask_previous[index_split_bit] = 0;

        // ds first we have to split the descriptors by the found index - preallocate vectors since
        // we know how many ones we have
        MatchableVector matchables_ones(number_of_on_bits_total);
        MatchableVector matchables_zeros(matchables.size() - number_of_on_bits_total);

        // ds loop over all descriptors and assigning them to the new vectors based on bit status
        uint64_t index_ones  = 0;
        uint64_t index_zeros = 0;
        for (Matchable* matchable : matchables) {
          if (matchable->descriptor[index_split_bit]) {
            matchables_ones[index_ones] = matchable;
            ++index_ones;
          } else {
            matchables_zeros[index_zeros] = matchable;
            ++index_zeros;
          }
        }
        assert(matchables_ones.size() == index_ones);
        assert(matchables_zeros.size() == index_zeros);

        // ds this leaf becomes a regular node and hence does not carry matchables
        has_leafs = true;
        matchables.clear();
        _header.number_of_matchables_compressed = 0;

        // ds if there are elements for leaves
        assert(0 < matchables_ones.size());
        right = new Node(this, _header.depth + 1, matchables_ones, bit_mask_previous, train_mode_);

        assert(0 < matchables_zeros.size());
        left = new Node(this, _header.depth + 1, matchables_zeros, bit_mask_previous, train_mode_);

        // ds success
        return true;
      } else {
        // ds failed to spawn leaf - terminate recursion
        return false;
      }
    }

    // ds getters
  public:
    const MatchableVector& getMatchables() const {
      return matchables;
    }
    const uint64_t& getDepth() const {
      return _header.depth;
    }
    const int32_t& indexSplitBit() const {
      return index_split_bit;
    }
    const uint64_t& getNumberOfSetBits() const {
      return number_of_on_bits_total;
    }
    const bool& hasLeafs() const {
      return has_leafs;
    }

    // ds inner constructors (used for recursive tree building)
  protected:
    // ds only internally called: default for single matchables
    BinaryNode(Node* parent_,
               const uint64_t& depth_,
               const MatchableVector& matchables_,
               Descriptor bit_mask_,
               const SplittingStrategy& train_mode_) :
      parent(parent_),
      _header(depth_),
      matchables(matchables_),
      bit_mask(bit_mask_) {
#ifdef SRRG_MERGE_DESCRIPTORS
      // ds recompute current number of contained merged matchables TODO make this less horribly
      // wasteful
      _header.number_of_matchables_uncompressed = 0;
      for (const Matchable* matchable : matchables) {
        _header.number_of_matchables_uncompressed += matchable->number_of_objects;
      }
#else
      _header.number_of_matchables_uncompressed = matchables.size();
#endif
      spawnLeafs(train_mode_);
    }

    // ds helpers
  protected:
    const real_type _getSetBitFraction(const uint32_t& index_split_bit_,
                                       const MatchableVector& matchables_,
                                       uint64_t& number_of_set_bits_total_) const {
      assert(0 < matchables_.size());
      assert(0 < _header.number_of_matchables_uncompressed);
      assert(matchables_.size() <= _header.number_of_matchables_uncompressed);

      // ds count set bits of all matchables in this node
      uint64_t number_of_set_bits = 0;
#ifdef SRRG_MERGE_DESCRIPTORS
      uint64_t number_of_set_bits_actual = 0;
      for (const Matchable* matchable : matchables_) {
        // ds accumulate set bit matchable counts
        if (matchable->descriptor[index_split_bit_]) {
          // ds make sure to weight merged matchables! default is 1, if not merged
          number_of_set_bits += matchable->number_of_objects;
          ++number_of_set_bits_actual;
        }
      }

      // ds this number is used to allocate for the partitioning
      // ds so we cannot count merged matchables twice here
      number_of_set_bits_total_ = number_of_set_bits_actual;
#else
      for (const Matchable* matchable : matchables_) {
        // ds just add the bits up (a set counts automatically as one)
        number_of_set_bits += matchable->descriptor[index_split_bit_];
      }
      number_of_set_bits_total_ = number_of_set_bits;
#endif
      assert(number_of_set_bits <= _header.number_of_matchables_uncompressed);

      // ds return ratio
      return (static_cast<real_type>(number_of_set_bits) /
              _header.number_of_matchables_uncompressed);
    }

    // ds public fields
  public:
    //! @brief leaf containing all unset bits
    Node* left = nullptr;

    //! @brief leaf containing all set bits
    Node* right = nullptr;

    //! @brief parent node (if any, for root:parent=0)
    Node* parent = nullptr;

    //! @brief minimum number of matchables in a node before splitting
    static uint64_t maximum_leaf_size;

    //! @brief maximum achieved descriptor group partitioning using the index_split_bit
    static real_type maximum_partitioning;

    //! @brief maximum tree depth (leaf spawning blocks if reached, default: descriptor dimension)
    static uint32_t maximum_depth;

    // ds fields
  protected:
    //! @brief serializable header carrying core attributes
    Header _header;

    //! @brief matchables contained in this node
    MatchableVector matchables;

    //! @brief the split bit diving potential leafs of this node
    int32_t index_split_bit = -1;

    //! @brief number of bits with value on
    uint64_t number_of_on_bits_total = 0;

    //! @brief flag set if the current node has 2 leafs
    bool has_leafs = false;

    //! @brief achieved descriptor group partitioning using the index_split_bit
    real_type partitioning = 1;

    //! @brief bit splitting mask considered before choosing index_split_bit
    Descriptor bit_mask;

    // ds random number generator, used for random splitting (for all nodes)
    static std::mt19937 random_number_generator;

    //! @brief allow direct access for processing classes
    template <typename BinaryNodeType_>
    friend class BinaryTree;
  };

  // ds default configuration
  template <typename BinaryMatchableType_, typename real_type_>
  uint64_t BinaryNode<BinaryMatchableType_, real_type_>::maximum_leaf_size = 100;
  template <typename BinaryMatchableType_, typename real_type_>
  real_type_ BinaryNode<BinaryMatchableType_, real_type_>::maximum_partitioning = 0.1;
  template <typename BinaryMatchableType_, typename real_type_>
  uint32_t BinaryNode<BinaryMatchableType_, real_type_>::maximum_depth =
    BinaryMatchableType_::descriptor_size_bits;
  template <typename BinaryMatchableType_, typename real_type_>
  std::mt19937 BinaryNode<BinaryMatchableType_, real_type_>::random_number_generator;

  template <typename ObjectType_>
  using BinaryNode128 = BinaryNode<BinaryMatchable128<ObjectType_>>;
  template <typename ObjectType_>
  using BinaryNode256 = BinaryNode<BinaryMatchable256<ObjectType_>>;
  template <typename ObjectType_>
  using BinaryNode512 = BinaryNode<BinaryMatchable512<ObjectType_>>;

} // namespace srrg_hbst
