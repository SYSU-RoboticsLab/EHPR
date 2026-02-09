#pragma once
#include "probabilistic_matchable.hpp"
#include "srrg_hbst/types/binary_node.hpp"

namespace srrg_hbst {

  // ds Node template specialization for descriptors of type: CDescriptorBinaryProbabilistic
  template <typename ProbabilisticMatchableType_, typename real_precision_ = double>
  class ProbabilisticNode : public BinaryNode<ProbabilisticMatchableType_, real_precision_> {
    // ds readability
    using Node = ProbabilisticNode<ProbabilisticMatchableType_, real_precision_>;

    // ds template forwarding
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef BinaryNode<ProbabilisticMatchableType_, real_precision_> BaseNode;
    typedef ProbabilisticMatchableType_ Matchable;
    typedef typename Matchable::Descriptor Descriptor;
    typedef std::vector<Matchable*> MatchableVector;
    typedef real_precision_ precision;
    typedef BinaryMatch<Matchable, precision> Match;
    typedef typename Matchable::BitStatisticsVector BitStatisticsVector;

    // ds ctor/dtor
  public:
    // ds access only through this constructor: no mask provided
    ProbabilisticNode(const MatchableVector& matchables_,
                      const SplittingStrategy& train_mode_ = SplittingStrategy::SplitEven) :
      Node(0, matchables_, Descriptor().set(), train_mode_) {
    }

    // ds the default constructor is triggered by subclasses - the responsibility of attribute
    // initialization is left to the subclass ds the default constructor is required, since we do not
    // want to trigger the automatic leaf spawning of the baseclass in a subclass
    ProbabilisticNode() {
    }

    // ds destructor: nothing to do (the leafs will be freed by the tree)
    virtual ~ProbabilisticNode() {
    }

    // ds access
  public:
    // ds implementing the new spawn leafs function to handle the modified descriptors
    virtual const bool spawnLeafs(const SplittingStrategy& train_mode_) override {
      // ds buffer number of descriptors
      const uint64_t number_of_matchables = this->matchables.size();

      // ds if there are at least 2 descriptors (minimal split)
      if (1 < number_of_matchables) {
        assert(!this->has_leafs);

        // ds affirm initial situation
        this->index_split_bit            = -1;
        this->number_of_on_bits_total    = 0;
        this->partitioning               = 1;
        real_precision_ variance_maximum = 0;

        // ds variance computation statistics
        BitStatisticsVector bit_probabilities_accumulated(BitStatisticsVector::Zero());

        // ds for all descriptors in this node (use base class descriptor)
        for (const Matchable* matchable : this->matchables) {
          // ds reduce accumulated probabilities (cast to the active descriptor)
          bit_probabilities_accumulated += matchable->bit_probabilities;
        }

        // ds get average
        const BitStatisticsVector bit_probabilities_mean(bit_probabilities_accumulated /
                                                         number_of_matchables);

        // ds compute variance
        for (uint32_t index_bit = 0; index_bit < Matchable::descriptor_size_bits; ++index_bit) {
          // ds if this index is available in the mask
          if (this->bit_mask[index_bit]) {
            // ds buffers
            real_precision_ variance_current = 0.0;

            // ds for all descriptors in this node
            for (const Matchable* matchable : this->matchables) {
              // ds update variance value
              const real_precision_ delta =
                matchable->bit_probabilities[index_bit] - bit_probabilities_mean[index_bit];
              variance_current += delta * delta;
            }

            // ds average
            variance_current /= number_of_matchables;

            // ds check if better
            if (variance_maximum < variance_current) {
              variance_maximum      = variance_current;
              this->index_split_bit = index_bit;
            }
          }
        }

        // ds if best was found - we can spawn leaves
        if (-1 != this->index_split_bit) {
          // ds compute distance for this index (0.0 is perfect)
          this->partitioning = std::fabs(
            0.5 - this->_getSetBitFraction(
                    this->index_split_bit, this->matchables, this->number_of_on_bits_total));

          // ds check if we have enough data to split
          if (0 < this->number_of_on_bits_total && 0.5 > this->partitioning) {
            // ds enabled
            this->has_leafs = true;

            // ds get a mask copy
            Descriptor mask(this->bit_mask);

            // ds update mask for leafs
            mask[this->index_split_bit] = 0;

            // ds first we have to split the descriptors by the found index - preallocate vectors
            // since we know how many ones we have
            MatchableVector matchables_leaf_ones;
            matchables_leaf_ones.reserve(this->number_of_on_bits_total);
            MatchableVector matchables_leaf_zeroes;
            matchables_leaf_zeroes.reserve(number_of_matchables - this->number_of_on_bits_total);

            // ds loop over all descriptors and assing them to the new vectors
            for (Matchable* matchable : this->matchables) {
              // ds check if split bit is set
              if (matchable->descriptor[this->index_split_bit]) {
                matchables_leaf_ones.push_back(matchable);
              } else {
                matchables_leaf_zeroes.push_back(matchable);
              }
            }

            // ds if there are elements for leaves
            assert(0 < matchables_leaf_ones.size());
            this->right = new Node(this->depth + 1, matchables_leaf_ones, mask, train_mode_);

            assert(0 < matchables_leaf_zeroes.size());
            this->left = new Node(this->depth + 1, matchables_leaf_zeroes, mask, train_mode_);

            // ds worked
            return true;
          } else {
            // ds split failed
            return false;
          }
        } else {
          // ds split failed
          return false;
        }
      } else {
        // ds not enough descriptors to split
        return false;
      }
    }

    // ds inner constructors (used for recursive tree building)
  protected:
    // ds only internally called: without split order
    ProbabilisticNode(const uint64_t& depth_,
                      const MatchableVector& matchables_,
                      Descriptor bit_mask_,
                      const SplittingStrategy& train_mode_) {
      // ds initialize subclass fields
      this->depth      = depth_;
      this->matchables = matchables_;
      this->has_leafs  = false;
      this->bit_mask   = bit_mask_;
      this->right      = 0;
      this->left       = 0;

      // ds call recursive leaf spawner
      spawnLeafs(train_mode_);
    }
  };
} // namespace srrg_hbst
