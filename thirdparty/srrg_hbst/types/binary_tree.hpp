#pragma once
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>

#include "binary_node.hpp"

// ds helper macro for controlled reading and writing operations
#define GUARDED_IO(FILE, IO_OPERATION, VARIABLE, SIZE, ERROR_MESSAGE) \
  if (!FILE.IO_OPERATION(VARIABLE, SIZE)) {                           \
    std::cerr << ERROR_MESSAGE << std::endl;                          \
    FILE.close();                                                     \
    return false;                                                     \
  }

namespace srrg_hbst {

  //! @class the binary tree class, consisting of binary nodes holding binary descriptors
  template <typename BinaryNodeType_>
  class BinaryTree {
    // ds exports
  public:
    //! @brief directly exported types (readability)
    using Node                  = typename BinaryNodeType_::BaseNode;
    using Matchable             = typename Node::Matchable;
    using MatchableVector       = typename Node::MatchableVector;
    using Descriptor            = typename Node::Descriptor;
    using Match                 = typename Node::Match;
    using real_type             = typename Node::real_type;
    using ObjectType            = typename Matchable::ObjectType;
    using ObjectMap             = typename Matchable::ObjectMap;
    using ObjectMapElement      = std::pair<uint64_t, ObjectType>;
    using MatchVector           = std::vector<Match>;
    using MatchVectorMap        = std::unordered_map<uint64_t, std::vector<Match>>;
    using MatchVectorMapElement = std::pair<uint64_t, std::vector<Match>>;

#ifdef SRRG_MERGE_DESCRIPTORS
    //! @brief component object used for matchable merging
    struct MatchableMerge {
      MatchableMerge(const Matchable* query_, ObjectType query_object_, Matchable* reference_) :
        query(query_),
        query_object(query_object_),
        reference(reference_) {
      }
      const Matchable* query;  // to be absorbed matchable
      ObjectType query_object; // affected object that might have to update its matchable reference
      Matchable* reference;    // absorbing matchable
    };
    typedef std::vector<MatchableMerge> MatchableMergeVector;
#endif

    //! @brief component object used for tree training
    struct Trainable {
      Node* node           = nullptr;
      Matchable* matchable = nullptr;
    };

    //! @brief image score for a reference image (added)
    struct Score {
      uint64_t number_of_matches    = 0;
      real_type matching_ratio      = 0;
      uint64_t identifier_reference = 0;
    };
    typedef std::vector<Score> ScoreVector;

    //! @brief object header containing main attributes
    struct Header {
      Header(const uint64_t& identifier_ = 0) :
        identifier(identifier_),
        number_of_matchables_compressed(0),
        number_of_training_entries(0),
        number_of_matchables_uncompressed(0),
        number_of_leafs(0) {
      }
      uint64_t identifier;
      uint64_t number_of_matchables_compressed;
      uint64_t number_of_training_entries;
      uint64_t number_of_matchables_uncompressed;
      uint64_t number_of_leafs;
#ifdef SRRG_MERGE_DESCRIPTORS
      static constexpr bool srrg_merge_descriptors = true;
#else
      static constexpr bool srrg_merge_descriptors = false;
#endif
      static constexpr size_t descriptor_size_bits = Matchable::descriptor_size_bits;
    };

    // ds ctor/dtor
  public:
    // ds empty tree instantiation with specific identifier
    BinaryTree(const uint64_t& identifier_) : _header(identifier_), _root(nullptr) {
      _matchables.clear();
      _matchables_to_train.clear();
      _added_identifiers_train.clear();
      _trainables.clear();
#ifdef SRRG_MERGE_DESCRIPTORS
      _merged_matchables.clear();
#endif
    }

    // ds empty tree instantiation
    BinaryTree() : BinaryTree(0) {
    }

    // ds construct tree upon allocation on filtered descriptors
    BinaryTree(const uint64_t& identifier_,
               const MatchableVector& matchables_,
               const SplittingStrategy& train_mode_ = SplittingStrategy::SplitEven) :
      _header(identifier_),
      _root(new Node(matchables_, train_mode_)) {
      _matchables.clear();
      _matchables.insert(_matchables.end(), matchables_.begin(), matchables_.end());
      _matchables_to_train.clear();
      _added_identifiers_train.clear();
      _added_identifiers_train.insert(_header.identifier);
      _trainables.clear();
#ifdef SRRG_MERGE_DESCRIPTORS
      _merged_matchables.clear();
#endif
    }

    // ds construct tree upon allocation on filtered descriptors
    BinaryTree(const MatchableVector& matchables_,
               const SplittingStrategy& train_mode_ = SplittingStrategy::SplitEven) :
      BinaryTree(0, matchables_, train_mode_) {
    }

    // ds construct tree upon allocation on filtered descriptors: with bit mask
    BinaryTree(const uint64_t& identifier_,
               const MatchableVector& matchables_,
               Descriptor bit_mask_,
               const SplittingStrategy& train_mode_ = SplittingStrategy::SplitEven) :
      _header(identifier_),
      _root(new Node(matchables_, bit_mask_, train_mode_)) {
      _matchables.clear();
      _matchables.insert(_matchables.end(), matchables_.begin(), matchables_.end());
      _matchables_to_train.clear();
      _added_identifiers_train.clear();
      _added_identifiers_train.insert(_header.identifier);
      _trainables.clear();
#ifdef SRRG_MERGE_DESCRIPTORS
      _merged_matchables.clear();
#endif
    }

    // ds free all nodes in the tree without freeing the matchables - call clear(true)
    ~BinaryTree() {
      clear();
    }

    // ds shared pointer access wrappers
  public:
    const uint64_t
    getNumberOfMatches(const std::shared_ptr<const MatchableVector> matchables_query_,
                       const uint32_t& maximum_distance_ = 25) const {
      return getNumberOfMatches(*matchables_query_, maximum_distance_);
    }

    const typename Node::real_type
    getMatchingRatio(const std::shared_ptr<const MatchableVector> matchables_query_,
                     const uint32_t& maximum_distance_ = 25) const {
      return getMatchingRatio(*matchables_query_, maximum_distance_);
    }

    const uint64_t
    getNumberOfMatchesLazy(const std::shared_ptr<const MatchableVector> matchables_query_,
                           const uint32_t& maximum_distance_ = 25) const {
      return getNumberOfMatchesLazy(*matchables_query_, maximum_distance_);
    }

    const typename Node::real_type
    getMatchingRatioLazy(const std::shared_ptr<const MatchableVector> matchables_query_,
                         const uint32_t& maximum_distance_ = 25) const {
      return getMatchingRatioLazy(*matchables_query_, maximum_distance_);
    }

    const typename Node::real_type
    getMatchingRatioLazy(const MatchableVector& matchables_query_,
                         const uint32_t& maximum_distance_ = 25) const {
      return static_cast<typename Node::real_type>(
               getNumberOfMatchesLazy(matchables_query_, maximum_distance_)) /
             matchables_query_.size();
    }

    const typename Node::real_type getMatchingRatio(const MatchableVector& matchables_query_,
                                                    const uint32_t& maximum_distance_ = 25) const {
      return static_cast<typename Node::real_type>(
               getNumberOfMatches(matchables_query_, maximum_distance_)) /
             matchables_query_.size();
    }

    //! @brief returns database size (i.e. the number of added images/matchable vectors with unique
    //! identifiers)
    //! @returns number of added reference matchable vectors
    const size_t size() const {
      assert(_header.number_of_training_entries == _added_identifiers_train.size());
      return _header.number_of_training_entries;
    }

    const std::set<uint64_t>& trainedIdentifiers() const {
      return _added_identifiers_train;
    }

    //! @brief returns the number of matchables added for training (before compression)
    //! @returns number of matchables
    const size_t numberOfMatchablesUncompressed() const {
      return _header.number_of_matchables_uncompressed;
    }

    //! @brief returns the actual number of stored matchable objects (after compression)
    //! this number is always equal to the uncompressed one if SRRG_MERGE_DESCRIPTORS is disabled
    //! @returns number of matchables
    const size_t numberOfMatchablesCompressed() const {
      return _header.number_of_matchables_compressed;
    }

    //! @brief const accessor to root node
    const Node* root() const {
      return _root;
    }

    //! number of merged matchables in last call
    const size_t numberOfMergedMatchablesLastTraining() const {
#ifdef SRRG_MERGE_DESCRIPTORS
      return _number_of_merged_matchables_last_training;
#else
      // ds always zero if not enabled
      return 0;
#endif
    }

#ifdef SRRG_MERGE_DESCRIPTORS
    //! @brief matchable merges from last match/add/train call - intended for external bookkeeping
    //! update only
    //! @returns a vector of effective merges between matchables (recall that the memory of
    //! Mergable.query is already freed)
    const MatchableMergeVector getMerges() const {
      return _merged_matchables;
    }
#endif

    const uint64_t getNumberOfMatches(const MatchableVector& matchables_query_,
                                      const uint32_t& maximum_distance_ = 25) const {
      if (matchables_query_.empty()) {
        return 0;
      }
      uint64_t number_of_matches = 0;

      // ds for each descriptor
      for (const Matchable* matchable_query : matchables_query_) {
        // ds traverse tree to find this descriptor
        const Node* node_current = _root;
        while (node_current) {
          // ds if this node has leaves (is splittable)
          if (node_current->has_leafs) {
            // ds check the split bit and go deeper
            if (matchable_query->descriptor[node_current->index_split_bit]) {
              node_current = node_current->right;
            } else {
              node_current = node_current->left;
            }
          } else {
            // ds check current descriptors in this node and exit
            for (const Matchable* matchable_reference : node_current->matchables) {
              if (maximum_distance_ > matchable_query->distance(matchable_reference)) {
                ++number_of_matches;
                break;
              }
            }
            break;
          }
        }
      }
      return number_of_matches;
    }

    const ScoreVector getScorePerImage(const MatchableVector& matchables_query_,
                                       const bool sort_output           = false,
                                       const uint32_t maximum_distance_ = 25) const {
      if (matchables_query_.empty()) {
        return ScoreVector(0);
      }
      ScoreVector scores_per_image(_added_identifiers_train.size());

      // ds identifier to vector index mapping - simultaneously initialize result vector
      std::map<uint64_t, uint64_t> mapping_identifier_image_to_score;
      for (const uint64_t& identifier_reference : _added_identifiers_train) {
        scores_per_image[mapping_identifier_image_to_score.size()].identifier_reference =
          identifier_reference;
        mapping_identifier_image_to_score.insert(
          std::make_pair(identifier_reference, mapping_identifier_image_to_score.size()));
      }

      // ds for each query descriptor
      for (const Matchable* matchable_query : matchables_query_) {
        // ds traverse tree to find this descriptor
        const Node* node_current = _root;
        while (node_current) {
          // ds if this node has leaves (is splittable)
          if (node_current->has_leafs) {
            // ds check the split bit and go deeper
            if (matchable_query->descriptor[node_current->index_split_bit]) {
              node_current = node_current->right;
            } else {
              node_current = node_current->left;
            }
          } else {
            // ds check current descriptors for each reference image in this node and exit
            std::set<uint64_t> matched_references;
            for (const Matchable* matchable_reference : node_current->matchables) {
              if (matchable_query->distance(matchable_reference) < maximum_distance_) {
#ifdef SRRG_MERGE_DESCRIPTORS
                for (const ObjectMapElement& object : matchable_reference->objects) {
                  const uint64_t& identifier_reference = object.first;
#else
                const uint64_t& identifier_reference = matchable_reference->_image_identifier;
#endif

                  // ds the query matchable can be matched only once to each reference image
                  if (matched_references.count(identifier_reference) == 0) {
                    ++scores_per_image[mapping_identifier_image_to_score.at(identifier_reference)]
                        .number_of_matches;
                    matched_references.insert(identifier_reference);
                  }
#ifdef SRRG_MERGE_DESCRIPTORS
                }
#endif
              }
            }
            break;
          }
        }
      }

      // ds compute relative scores
      const real_type number_of_query_descriptors = matchables_query_.size();
      for (Score& score : scores_per_image) {
        score.matching_ratio = score.number_of_matches / number_of_query_descriptors;
      }

      // ds if desired, sort in descending order by matching ratio
      if (sort_output) {
        std::sort(
          scores_per_image.begin(), scores_per_image.end(), [](const Score& a, const Score& b) {
            return a.matching_ratio > b.matching_ratio;
          });
      }
      return scores_per_image;
    }

    const uint64_t getNumberOfMatchesLazy(const MatchableVector& matchables_query_,
                                          const uint32_t& maximum_distance_ = 25) const {
      if (matchables_query_.empty()) {
        return 0;
      }
      uint64_t number_of_matches = 0;

      // ds for each descriptor
      for (const Matchable* matchable_query : matchables_query_) {
        // ds traverse tree to find this descriptor
        const Node* node_current = _root;
        while (node_current) {
          // ds if this node has leaves (is splittable)
          if (node_current->has_leafs) {
            // ds check the split bit and go deeper
            if (matchable_query->descriptor[node_current->index_split_bit]) {
              node_current = node_current->right;
            } else {
              node_current = node_current->left;
            }
          } else {
            // ds check current descriptors in this node and exit
            if (maximum_distance_ > matchable_query->distance(node_current->matchables.front())) {
              ++number_of_matches;
            }
            break;
          }
        }
      }
      return number_of_matches;
    }

    // ds direct matching function on this tree
    void matchLazy(const MatchableVector& matchables_query_,
                   MatchVector& matches_,
                   const uint32_t& maximum_distance_ = 25) const {
      if (matchables_query_.empty()) {
        return;
      }

      // ds for each descriptor
      for (const Matchable* matchable_query : matchables_query_) {
        // ds traverse tree to find this descriptor
        const Node* node_current = _root;
        while (node_current) {
          // ds if this node has leaves (is splittable)
          if (node_current->has_leafs) {
            // ds check the split bit and go deeper
            if (matchable_query->descriptor[node_current->index_split_bit]) {
              node_current = node_current->right;
            } else {
              node_current = node_current->left;
            }
          } else {
            // ds check current descriptors in this node and exit
            for (const Matchable* matchable_reference : node_current->matchables) {
              const real_type distance = matchable_query->distance(matchable_reference);
              if (distance < maximum_distance_) {
                matches_.push_back(Match(matchable_query,
                                         matchable_reference,
                                         matchable_query->objects.begin()->second,
                                         matchable_reference->objects.begin()->second,
                                         distance));
                break;
              }
            }
            break;
          }
        }
      }
    }

    // ds direct matching function on this tree
    void match(const MatchableVector& matchables_query_,
               MatchVector& matches_,
               const uint32_t& maximum_distance_ = 25) const {
      if (matchables_query_.empty()) {
        return;
      }

      // ds for each descriptor
      for (const Matchable* matchable_query : matchables_query_) {
        // ds traverse tree to find this descriptor
        const Node* node_current = _root;
        while (node_current) {
          // ds if this node has leaves (is splittable)
          if (node_current->has_leafs) {
            // ds check the split bit and go deeper
            if (matchable_query->descriptor[node_current->index_split_bit]) {
              node_current = node_current->right;
            } else {
              node_current = node_current->left;
            }
          } else {
            // ds current best (0 if none)
            const Matchable* matchable_reference_best = nullptr;
            uint32_t distance_best                    = maximum_distance_;

            // ds check current descriptors in this node and exit
            for (const Matchable* matchable_reference : node_current->matchables) {
              const uint32_t& distance = matchable_query->distance(matchable_reference);
              if (distance < distance_best) {
                matchable_reference_best = matchable_reference;
                distance_best            = distance;
              }
            }

            // ds if a match was found
            if (matchable_reference_best) {
              matches_.push_back(Match(matchable_query,
                                       matchable_reference_best,
                                       matchable_query->objects.begin()->second,
                                       matchable_reference_best->objects.begin()->second,
                                       distance_best));
            }
            break;
          }
        }
      }
    }

    // ds return matches directly
    const std::shared_ptr<const MatchVector>
    getMatchesLazy(const std::shared_ptr<const MatchableVector> matchables_query_,
                   const uint32_t& maximum_distance_ = 25) const {
      // ds wrap the call
      MatchVector matches;
      matchLazy(*matchables_query_, matches, maximum_distance_);
      return std::make_shared<const MatchVector>(matches);
    }

    //! @brief knn multi-matching function
    //! @param[in] matchables_query_ query matchables
    //! @param[out] matches_ output matching results: contains all available matches for all
    //! training images added to the tree
    //! @param[in] maximum_distance_ the maximum distance allowed for a positive match response
    void match(const MatchableVector& matchables_query_,
               MatchVectorMap& matches_,
               const uint32_t& maximum_distance_matching_ = 25) const {
      if (matchables_query_.empty() || _added_identifiers_train.empty()) {
        return;
      }

      // ds prepare match vector map for all ids in the tree
      matches_.clear();
      for (const uint64_t identifier_tree : _added_identifiers_train) {
        matches_.insert(std::make_pair(identifier_tree, MatchVector()));

        // ds preallocate space to speed up match addition
        matches_.at(identifier_tree).reserve(matchables_query_.size());
      }

      // ds for each descriptor
      for (const Matchable* matchable_query : matchables_query_) {
        // ds traverse tree to find this descriptor
        const Node* node_current = _root;
        while (node_current) {
          // ds if this node has leaves (is splittable)
          if (node_current->has_leafs) {
            // ds check the split bit and go deeper
            if (matchable_query->descriptor[node_current->index_split_bit]) {
              node_current = node_current->right;
            } else {
              node_current = node_current->left;
            }
          } else {
            // ds obtain best matches in the current leaf via brute-force search
            std::map<uint64_t, Match> best_matches;
            // std::cout<<" depth"<< node_current->getDepth();
            _matchExhaustive(
              matchable_query, node_current->matchables, maximum_distance_matching_, best_matches);

            // ds register all matches in the output structure
            for (const std::pair<uint64_t, Match> best_match : best_matches) {
              matches_.at(best_match.first).push_back(best_match.second);
            }
            break;
          }
        }
      }
    }

    //! @brief incrementally grows the tree
    //! @param[in] matchables_ new input matchables to integrate into the current tree (transferring
    //! the ownership!)
    //! @param[in] train_mode_ train_mode_
    void add(const MatchableVector& matchables_,
             const SplittingStrategy& train_mode_ = SplittingStrategy::DoNothing) {
      if (matchables_.empty()) {
        return;
      }

      // ds prepare bookkeeping for training
      assert(matchables_.front()->_image_identifier == matchables_.back()->_image_identifier);
      _added_identifiers_train.insert(matchables_.front()->_image_identifier);
      ++_header.number_of_training_entries;
      _matchables_to_train.insert(
        _matchables_to_train.end(), matchables_.begin(), matchables_.end());

      // ds train based on set matchables (no effect for do SplittingStrategy::DoNothing)
      train(train_mode_);
    }

    //! @brief train tree with current _trainable_matchables according to selected mode
    //! @param[in] train_mode_ desired training mode
    void train(const SplittingStrategy& train_mode_ = SplittingStrategy::SplitEven) {
      if (_matchables_to_train.empty() || train_mode_ == SplittingStrategy::DoNothing) {
        return;
      }
      _header.number_of_matchables_uncompressed += _matchables_to_train.size();

      // ds check if we have to build an initial tree first (no training afterwards)
      if (!_root) {
        _root = new Node(_matchables_to_train, train_mode_);
        assert(_matchables.empty());
        _matchables.insert(
          _matchables.end(), _matchables_to_train.begin(), _matchables_to_train.end());
        _header.number_of_matchables_compressed = _matchables_to_train.size();
        _matchables_to_train.clear();
        return;
      }

      // ds if random splitting is chosen
      if (train_mode_ == SplittingStrategy::SplitRandomUniform) {
        // ds initialize random number generator with new seed
        std::random_device random_device;
        Node::random_number_generator = std::mt19937(random_device());
      }

      // ds nodes to update after the addition of matchables to leafs
      std::set<Node*> leafs_to_update;

#ifdef SRRG_MERGE_DESCRIPTORS
      // ds we need delayed insertion as we continuously scan the current references for merging
      _trainables.resize(_matchables_to_train.size());

      // ds matches to merge (descriptor distance == SRRG_MERGE_DESCRIPTORS)
      _merged_matchables.clear();
      _merged_matchables.reserve(_matchables_to_train.size());

      // ds currently we allow merging maximally once per reference matchable
      std::set<const Matchable*> merged_reference_matchables;
#endif

      // ds for each new descriptor - buffering new matchables and merging identical ones
      uint64_t index_new_matchable = 0;
      for (Matchable* matchable_to_insert : _matchables_to_train) {
        // ds traverse tree to find a leaf for this descriptor
        Node* node_current = _root;
        while (node_current) {
          // ds if this node has leaves (traversable)
          if (node_current->has_leafs) {
            // ds check the split bit and traverse the tree
            if (matchable_to_insert->descriptor[node_current->index_split_bit]) {
              node_current = node_current->right;
            } else {
              node_current = node_current->left;
            }
          } else {
            // ds we arrived in a leaf
#ifdef SRRG_MERGE_DESCRIPTORS
            bool insertion_required = true;

            // ds if we can absorb this matchable instead of having to insert it
            for (const Matchable* matchable_reference : node_current->matchables) {
              // ds if merge distance is satisfied
              // ds and this reference has not absorbed a matchable already in this call
              if (matchable_reference->distance(matchable_to_insert) <=
                    maximum_distance_for_merge &&
                  merged_reference_matchables.count(matchable_reference) == 0) {
                assert(matchable_reference != matchable_to_insert);
                assert(matchable_to_insert->objects.size() == 1);
                _merged_matchables.emplace_back(
                  MatchableMerge(matchable_to_insert,
                                 std::move(matchable_to_insert->_object),
                                 const_cast<Matchable*>(matchable_reference)));
                merged_reference_matchables.insert(matchable_reference);
                insertion_required = false;
                break;
              }
            }

            // ds if insertion is required - we could not merge the query matchable
            if (insertion_required) {
              // ds bookkeep matchable for addition
              // ds we need to bookkeep the node in order to add it to its matchables later on
              _trainables[index_new_matchable].node      = node_current;
              _trainables[index_new_matchable].matchable = matchable_to_insert;
              _matchables_to_train[index_new_matchable]  = matchable_to_insert;
              ++index_new_matchable;
            }
#else
            // ds we can place the descriptor in the leaf on the spot
            node_current->matchables.push_back(matchable_to_insert);
            _matchables_to_train[index_new_matchable] = matchable_to_insert;
            ++index_new_matchable;
#endif
            // ds leaf always needs to be updated, merged or not
            ++node_current->_header.number_of_matchables_uncompressed;
            leafs_to_update.insert(node_current);
            break;
          }
        }
      }
      _matchables_to_train.resize(index_new_matchable);
#ifdef SRRG_MERGE_DESCRIPTORS

      // ds merge matchables
      for (MatchableMerge& mergable : _merged_matchables) {
        assert(mergable.reference != mergable.query);

        // ds perform merge
        mergable.reference->mergeSingle(mergable.query);

        // ds free query (!) recall that the tree takes ownership of the matchables
        delete mergable.query;
      }
      _number_of_merged_matchables_last_training = _merged_matchables.size();
      _merged_matchables.clear();

      // ds insert matchables into nodes
      _trainables.resize(index_new_matchable);
      assert(_matchables_to_train.size() == _trainables.size());
      for (const Trainable& trainable : _trainables) {
        trainable.node->matchables.push_back(trainable.matchable);
      }
#endif
      // ds check splits for touched leafs
      for (Node* leaf : leafs_to_update) {
        leaf->spawnLeafs(train_mode_);
      }

      // ds bookkeeping
      _matchables.insert(
        _matchables.end(), _matchables_to_train.begin(), _matchables_to_train.end());
      _header.number_of_matchables_compressed += _matchables_to_train.size();
      _matchables_to_train.clear();
    }

    //! @brief knn multi-matching function with simultaneous adding
    //! @param[in] matchables_ query matchables, which will also automatically be added to the tree
    //! (transferring the ownership!)
    //! @param[out] matches_ output matching results: contains all available matches for all
    //! training images added to the tree
    //! @param[in] maximum_distance_matching_ the maximum distance allowed for a positive match
    //! response
    void matchAndAdd(const MatchableVector& matchables_,
                     MatchVectorMap& matches_,
                     const uint32_t maximum_distance_matching_ = 25,
                     const SplittingStrategy& train_mode_      = SplittingStrategy::SplitEven) {
      if (matchables_.empty()) {
        return;
      }
      const uint64_t identifier_image_query = matchables_.front()->_image_identifier;

      // ds check if we have to build an initial tree first
      if (!_root) {
        _root = new Node(matchables_);
        assert(_matchables.empty());
        _matchables.insert(_matchables.end(), matchables_.begin(), matchables_.end());
        _header.number_of_matchables_compressed = matchables_.size();
        _added_identifiers_train.insert(identifier_image_query);
        assert(_added_identifiers_train.size() == 1);
        _header.number_of_training_entries = 1;
        return;
      }

      // ds prepare match vector map for all ids in the tree
      matches_.clear();
      for (const uint64_t identifier_tree : _added_identifiers_train) {
        matches_.insert(std::make_pair(identifier_tree, MatchVector()));

        // ds preallocate space to speed up match addition
        matches_.at(identifier_tree).reserve(matchables_.size());
      }

      // ds prepare node/matchable list to integrate
      _trainables.resize(matchables_.size());
      std::set<Node*> leafs_to_update;

#ifdef SRRG_MERGE_DESCRIPTORS
      // ds matches to merge (descriptor distance == SRRG_MERGE_DESCRIPTORS)
      _merged_matchables.clear();
      _merged_matchables.reserve(matchables_.size());

      // ds currently we allow merging maximally once per reference matchable
      std::set<const Matchable*> merged_reference_matchables;
#endif

      // ds for each descriptor
      uint64_t index_trainable = 0;
      for (Matchable* matchable_query : matchables_) {
        // ds traverse tree to find this descriptor
        Node* node_current = _root;
        while (node_current) {
          // ds if this node has leaves (is splittable)
          if (node_current->has_leafs) {
            // ds check the split bit and go deeper
            if (matchable_query->descriptor[node_current->index_split_bit]) {
              node_current = node_current->right;
            } else {
              node_current = node_current->left;
            }
          } else {
            // ds obtain best matches in the current leaf via brute-force search - bookkeeping
            // matches to merge (distance == 0)
            std::map<uint64_t, Match> best_matches;
#ifdef SRRG_MERGE_DESCRIPTORS
            Matchable* matchable_reference = nullptr;
            _matchExhaustive(matchable_query,
                             node_current->matchables,
                             maximum_distance_matching_,
                             best_matches,
                             matchable_reference);
#else
            _matchExhaustive(
              matchable_query, node_current->matchables, maximum_distance_matching_, best_matches);
#endif

            // ds register all matches in the output structure
            for (const std::pair<uint64_t, Match> best_match : best_matches) {
              matches_.at(best_match.first).push_back(best_match.second);
            }

#ifdef SRRG_MERGE_DESCRIPTORS
            // ds if we can merge the query matchable into the reference
            if (matchable_reference &&
                merged_reference_matchables.count(matchable_reference) == 0) {
              assert(matchable_query->objects.size() == 1);

              // ds bookkeep matchable for merge
              _merged_matchables.emplace_back(MatchableMerge(
                matchable_query, std::move(matchable_query->_object), matchable_reference));
              merged_reference_matchables.insert(matchable_reference);
            } else {
#endif
              // ds bookkeep matchable for addition
              _trainables[index_trainable].node      = node_current;
              _trainables[index_trainable].matchable = matchable_query;
              ++index_trainable;
#ifdef SRRG_MERGE_DESCRIPTORS
            }
#endif

            // ds leaf needs to be updated, merged or not
            ++node_current->_header.number_of_matchables_uncompressed;
            leafs_to_update.insert(node_current);
            break;
          }
        }
      }
      _trainables.resize(index_trainable);
#ifdef SRRG_MERGE_DESCRIPTORS

      // ds merge matchables
      for (MatchableMerge& mergable : _merged_matchables) {
        assert(mergable.reference != mergable.query);

        // ds perform merge
        mergable.reference->mergeSingle(mergable.query);

        // ds free query (!) recall that the tree takes ownership of the matchables
        delete mergable.query;
      }
      _number_of_merged_matchables_last_training = _merged_matchables.size();
      _merged_matchables.clear();
#endif

      // ds integrate new matchables: merge, add and spawn leaves if requested
      MatchableVector new_matchables;
      new_matchables.reserve(_trainables.size());
      for (const Trainable& trainable : _trainables) {
        trainable.node->matchables.push_back(trainable.matchable);
        new_matchables.emplace_back(trainable.matchable);
      }
      for (Node* leaf : leafs_to_update) {
        leaf->spawnLeafs(train_mode_);
      }

      // ds insert new matchables and identifier
      _matchables.insert(_matchables.end(), new_matchables.begin(), new_matchables.end());
      _header.number_of_matchables_compressed += new_matchables.size();
      _added_identifiers_train.insert(identifier_image_query);
      ++_header.number_of_training_entries;
    }

#ifdef SRRG_HBST_HAS_OPENCV

    // ds creates a matchable vector (pointers) from opencv descriptors - only available if OpenCV
    // is present on building system
    static const MatchableVector getMatchables(const cv::Mat& descriptors_cv_,
                                               const std::vector<ObjectType>& objects_,
                                               const uint64_t& identifier_tree_ = 0) {
      MatchableVector matchables(descriptors_cv_.rows);

      // ds copy raw data
      for (int64_t index_descriptor = 0; index_descriptor < descriptors_cv_.rows;
           ++index_descriptor) {
        matchables[index_descriptor] = new Matchable(
          objects_[index_descriptor], descriptors_cv_.row(index_descriptor), identifier_tree_);
      }
      return matchables;
    }

#endif

    //! @brief clears complete structure (corresponds to empty construction)
    void clear(const bool& delete_matchables_ = true) {
      // ds database identifier is not reset

      // ds clean internal bookkeeping
      _added_identifiers_train.clear();
      _trainables.clear();
      _header.number_of_matchables_uncompressed = 0;
      _header.number_of_matchables_compressed   = 0;
      _header.number_of_training_entries        = 0;
#ifdef SRRG_MERGE_DESCRIPTORS
      _number_of_merged_matchables_last_training = 0;
#endif

      // ds recursively delete all nodes
      delete _root;
      _root = nullptr;

      // ds ownership dependent
      if (delete_matchables_) {
        deleteMatchables();
      }
      _matchables.clear();
      _matchables_to_train.clear();
    }

    //! @brief free all matchables contained in the tree (destructor)
    void deleteMatchables() {
      for (const Matchable* matchable : _matchables) {
        delete matchable;
      }
      for (const Matchable* matchable : _matchables_to_train) {
        delete matchable;
      }
    }

    //! ds save complete database to disk
    bool write(const std::string& file_path) const {
      // ds open file (overwriting existing)
      std::ofstream outfile(file_path, std::ios::binary | std::ios::out);
      if (!outfile.is_open()) {
        std::cerr << "BinaryTree::write|ERROR: unable to open file: " << file_path << std::endl;
        return false;
      }
      if (!outfile.good()) {
        std::cerr << "BinaryTree::write|ERROR: file is not valid: " << file_path << std::endl;
        return false;
      }

      // ds traverse complete tree to obtain leaf information
      uint64_t number_of_leafs      = 0;
      uint64_t number_of_matchables = 0;
      std::vector<const Node*> leafs;
      _getLeafs(_root, number_of_leafs, number_of_matchables, leafs);
      assert(number_of_matchables == _matchables.size());

      // ds set endianness byte flag - when reading we will check for zero value
      const char endianness_check[] = {char(0)};
      GUARDED_IO(outfile, write, endianness_check, 1, "BinaryTree::write|ERROR: unable to write");

      // ds populate leaf count in header (currently only used for de/serialization)
      _header.number_of_leafs = number_of_leafs;
      GUARDED_IO(outfile,
                 write,
                 reinterpret_cast<const char*>(&_header),
                 sizeof(_header),
                 "BinaryTree::write|ERROR: unable to write database header");
      for (const uint64_t identifier : _added_identifiers_train) {
        GUARDED_IO(outfile,
                   write,
                   reinterpret_cast<const char*>(&identifier),
                   sizeof(identifier),
                   "BinaryTree::write|ERROR: unable to write identifiers train");
      }

      // ds write leaf information
      for (const Node* leaf : leafs) {
        assert(leaf->_header.depth > 0);
        assert(leaf->index_split_bit == -1);
        assert(leaf->has_leafs == false);
#ifndef SRRG_MERGE_DESCRIPTORS
        assert(leaf->_header.number_of_matchables_uncompressed ==
               leaf->_header.number_of_matchables_compressed);
#endif
        GUARDED_IO(outfile,
                   write,
                   reinterpret_cast<const char*>(&leaf->_header),
                   sizeof(leaf->_header),
                   "BinaryTree::write|ERROR: unable to write leaf header");

        // ds store split bit index order
        const Node* current = leaf;
        std::vector<int32_t> indices_split_bit;
        indices_split_bit.reserve(leaf->_header.depth);
        while (current) {
          indices_split_bit.emplace_back(current->index_split_bit);
          current = current->parent;
        }
        for (auto it = indices_split_bit.rbegin(); it != indices_split_bit.rend(); ++it) {
          GUARDED_IO(outfile,
                     write,
                     reinterpret_cast<const char*>(&*it),
                     sizeof(int32_t),
                     "BinaryTree::write|ERROR: unable to write bit index order");
        }

        // ds serialize matchables (descriptor) data
        assert(leaf->_header.number_of_matchables_uncompressed >= leaf->matchables.size());
        for (const Matchable* matchable : leaf->matchables) {
          GUARDED_IO(outfile,
                     write,
                     reinterpret_cast<const char*>(&matchable->descriptor),
                     Matchable::raw_descriptor_size_bytes,
                     "BinaryTree::write|ERROR: unable to write descriptor data");
          GUARDED_IO(outfile,
                     write,
                     reinterpret_cast<const char*>(&matchable->number_of_objects),
                     sizeof(uint64_t),
                     "BinaryTree::write|ERROR: unable to write number of objects");
          assert(matchable->number_of_objects == matchable->objects.size());
          for (const ObjectMapElement& element : matchable->objects) {
            GUARDED_IO(outfile,
                       write,
                       reinterpret_cast<const char*>(&element.first),
                       sizeof(uint64_t),
                       "BinaryTree::write|ERROR: unable to write object key");
            GUARDED_IO(outfile,
                       write,
                       reinterpret_cast<const char*>(&element.second),
                       sizeof(ObjectType),
                       "BinaryTree::write|ERROR: unable to write object data");
          }
        }
      }
      outfile.close();
      return true;
    }

    //! ds load database from disk
    bool read(const std::string& file_path) {
      // ds open file for reading
      std::ifstream infile(file_path, std::ios::binary);
      if (!infile.is_open()) {
        std::cerr << "BinaryTree::read|ERROR: unable to open file: " << file_path << std::endl;
        return false;
      }
      if (!infile.good()) {
        std::cerr << "BinaryTree::read|ERROR: file is not valid: " << file_path << std::endl;
        return false;
      }

      // ds check endianness consistency
      char endianness_check[] = {char(1)};
      GUARDED_IO(
        infile, read, endianness_check, 1, "BinaryTree::read|ERROR: unable to read endian byte");
      if (endianness_check[0] != char(0)) {
        std::cerr << "BinaryTree::read|ERROR: invalid endianness, database saved on different arch"
                  << std::endl;
        infile.close();
        return false;
      }

      // ds read database header
      GUARDED_IO(infile, read, reinterpret_cast<char*>(&_header), sizeof(_header), "");
      for (size_t i = 0; i < _header.number_of_training_entries; ++i) {
        uint64_t identifier = 0;
        GUARDED_IO(infile, read, reinterpret_cast<char*>(&identifier), sizeof(identifier), "");
        _added_identifiers_train.insert(identifier);
      }
      assert(_added_identifiers_train.size() == _header.number_of_training_entries);

      // ds leaf buffer (static to allow easy escapes without requiring deallocation)
      // ds TODO use more compact data structure(s)
      std::vector<typename Node::Header> leaf_headers;
      std::vector<std::vector<Descriptor>> descriptors_per_leaf;
      std::vector<std::vector<ObjectMap>> objects_per_descriptor_per_leaf;
      std::vector<std::vector<int32_t>> bit_indexes_per_leaf;
      leaf_headers.reserve(_header.number_of_leafs);
      descriptors_per_leaf.reserve(_header.number_of_leafs);
      bit_indexes_per_leaf.reserve(_header.number_of_leafs);

      // ds read leafs with matchable data
      size_t number_of_read_matchables = 0;
      for (size_t i = 0; i < _header.number_of_leafs; ++i) {
        typename Node::Header leaf_header;
        GUARDED_IO(infile,
                   read,
                   reinterpret_cast<char*>(&leaf_header),
                   sizeof(leaf_header),
                   "BinaryTree::read|ERROR: unable to read Node header");
        assert(leaf_header.depth > 0);
        assert(leaf_header.number_of_matchables_uncompressed > 0);
#ifdef SRRG_MERGE_DESCRIPTORS
        assert(leaf_header.number_of_matchables_compressed <=
               leaf_header.number_of_matchables_uncompressed);
#else
        assert(leaf_header.number_of_matchables_uncompressed ==
               leaf_header.number_of_matchables_compressed);
#endif

        // ds read split bit indices order - note that we also have to read the -1 of the leaf
        std::vector<int32_t> indices_split_bit(leaf_header.depth + 1);
        for (size_t j = 0; j < leaf_header.depth + 1; ++j) {
          GUARDED_IO(infile,
                     read,
                     reinterpret_cast<char*>(&indices_split_bit[j]),
                     sizeof(int32_t),
                     "BinaryTree::read|ERROR: unable to read bit index order");
        }
        assert(indices_split_bit.back() == -1);

        // ds read matchables of this leaf
        std::vector<Descriptor> descriptors;
        std::vector<ObjectMap> objects_per_descriptor;
        descriptors.reserve(leaf_header.number_of_matchables_compressed);
        objects_per_descriptor.reserve(leaf_header.number_of_matchables_compressed);
        for (size_t j = 0; j < leaf_header.number_of_matchables_compressed; ++j) {
          Descriptor descriptor;
          GUARDED_IO(infile,
                     read,
                     reinterpret_cast<char*>(&descriptor),
                     Matchable::raw_descriptor_size_bytes,
                     "BinaryTree::read|ERROR: unable to read Matchable data");
          descriptors.emplace_back(descriptor);
          uint64_t number_of_objects = 0;
          GUARDED_IO(infile,
                     read,
                     reinterpret_cast<char*>(&number_of_objects),
                     sizeof(uint64_t),
                     "BinaryTree::read|ERROR: unable to read number of objects");
          ObjectMap objects;
          for (size_t index_object = 0; index_object < number_of_objects; ++index_object) {
            uint64_t key = 0;
            ObjectType object;
            GUARDED_IO(infile,
                       read,
                       reinterpret_cast<char*>(&key),
                       sizeof(uint64_t),
                       "BinaryTree::read|ERROR: unable to read object key");
            GUARDED_IO(infile,
                       read,
                       reinterpret_cast<char*>(&object),
                       sizeof(ObjectType),
                       "BinaryTree::read|ERROR: unable to read object");
            objects.insert(std::make_pair(key, object));
          }
          objects_per_descriptor.emplace_back(objects);
        }
        number_of_read_matchables += descriptors.size();
        leaf_headers.emplace_back(leaf_header);
        descriptors_per_leaf.emplace_back(descriptors);
        objects_per_descriptor_per_leaf.emplace_back(objects_per_descriptor);
        bit_indexes_per_leaf.emplace_back(indices_split_bit);
      }
      infile.close();

      // ds consistency check
      if (number_of_read_matchables != _header.number_of_matchables_compressed) {
        std::cerr << "BinaryTree::read|ERROR: number of loaded matchables inconsistent with header"
                  << std::endl;
        return false;
      }

      // ds after this point we use dynamic memory to build the tree - no exceptions are thrown!
      // ds assemble actual database by evaluating all leafs
      _root = new Node();
      assert(leaf_headers.size() == bit_indexes_per_leaf.size());
      for (size_t i = 0; i < leaf_headers.size(); ++i) {
        const typename Node::Header& leaf_header             = leaf_headers[i];
        const std::vector<Descriptor>& descriptors           = descriptors_per_leaf[i];
        const std::vector<int32_t>& bit_index_order          = bit_indexes_per_leaf[i];
        const std::vector<ObjectMap>& objects_per_descriptor = objects_per_descriptor_per_leaf[i];

        // ds grab any descriptor for evaluation decision rule
        // ds this is fine since all the descriptors reside in the same leaf and satisfy that rule
        const Descriptor& descriptor_sample = descriptors.back();

        // ds start from root for each leaf
        Node* current = _root;
        while (current) {
          // ds terminate if we reached a leaf
          if (bit_index_order[current->_header.depth] == -1) {
            assert(current->_header.depth == leaf_header.depth);
            assert(descriptors.size() == leaf_header.number_of_matchables_compressed);

            // ds populate matchables of the leaf
            assert(current->matchables.empty());
            current->matchables.reserve(descriptors.size());
            for (size_t index_descriptor = 0; index_descriptor < descriptors.size();
                 ++index_descriptor) {
              current->matchables.emplace_back(new Matchable(
                objects_per_descriptor[index_descriptor], descriptors[index_descriptor]));
            }
            current->_header = std::move(leaf_header);
            _matchables.insert(
              _matchables.end(), current->matchables.begin(), current->matchables.end());
            break;
          } else {
            // ds otherwise it is always an intermediate node
            current->has_leafs = true;
          }

          // ds if this node has not been initialized yet
          if (current->index_split_bit != bit_index_order[current->_header.depth]) {
            current->index_split_bit                    = bit_index_order[current->_header.depth];
            current->bit_mask[current->index_split_bit] = 0;
          }

          // ds spawn leafs if necessary (we have a complete tree)
          if (!current->right) {
            current->right                = new Node();
            current->right->_header.depth = current->_header.depth + 1;
            current->right->parent        = current;
          }
          if (!current->left) {
            current->left                = new Node();
            current->left->_header.depth = current->_header.depth + 1;
            current->left->parent        = current;
          }

          // ds traverse tree
          if (descriptor_sample[current->index_split_bit]) {
            current = current->right;
          } else {
            current = current->left;
          }
        }
      }

      // ds consistency check
      if (_matchables.size() != _header.number_of_matchables_compressed) {
        std::cerr << "BinaryTree::read|ERROR: unable to reconstruct tree with read matchables "
                     "(check HBST version used to generate database)"
                  << std::endl;
        return false;
      }
      return true;
    }

    // ds helpers
  protected:
#ifdef SRRG_MERGE_DESCRIPTORS
    //! @brief retrieves best matches (BF search) for provided matchables for all image indices
    //! @param[in] matchable_query_
    //! @param[in] matchables_reference_
    //! @param[in] maximum_distance_matching_
    //! @param[in,out] best_matches_ best match search storage: image id, match candidate
    void _matchExhaustive(const Matchable* matchable_query_,
                          const MatchableVector& matchables_reference_,
                          const uint32_t& maximum_distance_matching_,
                          std::map<uint64_t, Match>& best_matches_) const {
      ObjectType object_query =
        std::move(matchable_query_->objects.at(matchable_query_->_image_identifier));

      // ds check current descriptors in this node
      for (const Matchable* matchable_reference : matchables_reference_) {
        // ds compute the descriptor distance
        const uint32_t distance = matchable_query_->distance(matchable_reference);

        // ds if matching distance is within the threshold
        if (distance < maximum_distance_matching_) {
          // ds for every reference in this matchable
          for (const ObjectMapElement& object : matchable_reference->objects) {
            const uint64_t& identifer_tree_reference = object.first;

            try {
              // ds update match if current is better than the current best - will jump to addition
              // if no best is available
              Match& best_match_so_far              = best_matches_.at(identifer_tree_reference);
              const real_type& best_distance_so_far = best_match_so_far.distance;
              assert(best_distance_so_far >= 0);
              if (distance < best_distance_so_far) {
                assert(best_match_so_far.matchable_references.size() >= 1);
                assert(best_match_so_far.object_references.size() >= 1);

                // ds replace the best with this match on the spot - we don't have to update the
                // query information
                best_match_so_far.matchable_references =
                  std::move(std::vector<const Matchable*>(1, matchable_reference));
                best_match_so_far.object_references =
                  std::move(std::vector<ObjectType>(1, object.second));
                best_match_so_far.distance = distance;
                assert(best_match_so_far.matchable_references.size() == 1);
                assert(best_match_so_far.object_references.size() == 1);
              }

              // ds if the match is equal to the last (multiple candidates)
              else if (distance == best_distance_so_far) {
                // ds add the candidate - we don't have to update the distance since its the same as
                // the best
                best_match_so_far.matchable_references.push_back(matchable_reference);
                best_match_so_far.object_references.push_back(object.second);
                assert(best_match_so_far.matchable_references.size() > 1);
                assert(best_match_so_far.object_references.size() > 1);
              }
            } catch (const std::out_of_range& /*exception*/) {
              // ds add a new match
              best_matches_.insert(std::make_pair(
                identifer_tree_reference,
                Match(
                  matchable_query_, matchable_reference, object_query, object.second, distance)));
            }
          }
        }
      }
    }

    //! @brief retrieves best matches (BF search) for provided matchables for all image indices
    //! @param[in] matchable_query_
    //! @param[in] matchables_reference_
    //! @param[in] maximum_distance_matching_
    //! @param[in,out] best_matches_ best match search storage: image id, match candidate
    //! @param[in,out] matchable_reference_for_merge_ reference matchable with distance == 0
    //! (matchable merge candidate)
    void _matchExhaustive(const Matchable* matchable_query_,
                          const MatchableVector& matchables_reference_,
                          const uint32_t& maximum_distance_matching_,
                          std::map<uint64_t, Match>& best_matches_,
                          Matchable*& matchable_reference_for_merge_) const {
      ObjectType object_query =
        std::move(matchable_query_->objects.at(matchable_query_->_image_identifier));

      // ds check current descriptors in this node
      for (const Matchable* matchable_reference : matchables_reference_) {
        // ds compute the descriptor distance
        const uint32_t distance = matchable_query_->distance(matchable_reference);

        // ds if matching distance is within the threshold
        if (distance < maximum_distance_matching_) {
          // ds for every reference in this matchable
          for (const ObjectMapElement& object : matchable_reference->objects) {
            const uint64_t& identifer_tree_reference = object.first;

            try {
              // ds update match if current is better than the current best - will jump to addition
              // if no best is available
              Match& best_match_so_far              = best_matches_.at(identifer_tree_reference);
              const real_type& best_distance_so_far = best_match_so_far.distance;
              assert(best_distance_so_far >= 0);
              if (distance < best_distance_so_far) {
                assert(best_match_so_far.matchable_references.size() >= 1);
                assert(best_match_so_far.object_references.size() >= 1);

                // ds replace the best with this match on the spot - we don't have to update the
                // query information
                best_match_so_far.matchable_references =
                  std::move(std::vector<const Matchable*>(1, matchable_reference));
                best_match_so_far.object_references =
                  std::move(std::vector<ObjectType>(1, object.second));
                best_match_so_far.distance = distance;
                assert(best_match_so_far.matchable_references.size() == 1);
                assert(best_match_so_far.object_references.size() == 1);
              }

              // ds if the match is equal to the last (multiple candidates)
              else if (distance == best_distance_so_far) {
                // ds add the candidate - we don't have to update the distance since its the same as
                // the best
                best_match_so_far.matchable_references.push_back(matchable_reference);
                best_match_so_far.object_references.push_back(object.second);
                assert(best_match_so_far.matchable_references.size() > 1);
                assert(best_match_so_far.object_references.size() > 1);
              }
            } catch (const std::out_of_range& /*exception*/) {
              // ds add a new match
              best_matches_.insert(std::make_pair(
                identifer_tree_reference,
                Match(
                  matchable_query_, matchable_reference, object_query, object.second, distance)));
            }
          }

          // ds if the matchable descriptors are identical - we can merge - note that
          // maximum_distance_for_merge must always be smaller than maximum_distance_matching_
          assert(maximum_distance_for_merge < maximum_distance_matching_);
          if (distance <= maximum_distance_for_merge) {
            // ds behold the power of C++ (we want to keep the MatchableVector elements const)
            matchable_reference_for_merge_ = const_cast<Matchable*>(matchable_reference);
          }
        }
      }
    }
#else
    //! @brief retrieves best matches (BF search) for provided matchables for all image indices
    //! @param[in] matchable_query_
    //! @param[in] matchables_reference_
    //! @param[in] maximum_distance_matching_
    //! @param[in,out] best_matches_ best match search storage: image id, match candidate
    void _matchExhaustive(const Matchable* matchable_query_,
                          const MatchableVector& matchables_reference_,
                          const uint32_t& maximum_distance_matching_,
                          std::map<uint64_t, Match>& best_matches_) const {
      ObjectType object_query =
        std::move(matchable_query_->objects.at(matchable_query_->_image_identifier));

      // ds check current descriptors in this node
      for (const Matchable* matchable_reference : matchables_reference_) {
        // ds compute the descriptor distance
        const uint32_t distance = matchable_query_->distance(matchable_reference);

        // ds if matching distance is within the threshold
        if (distance < maximum_distance_matching_) {
          const uint64_t& identifer_tree_reference = matchable_reference->_image_identifier;
          assert(matchable_reference->objects.find(identifer_tree_reference) !=
                 matchable_reference->objects.end());
          ObjectType object_reference =
            std::move(matchable_reference->objects.at(identifer_tree_reference));

          try {
            // ds update match if current is better than the current best - will jump to addition if
            // no best is available
            Match& best_match_so_far = best_matches_.at(identifer_tree_reference);
            const real_type& best_distance_so_far = best_match_so_far.distance;
            assert(best_distance_so_far >= 0);
            if (distance < best_distance_so_far) {
              assert(best_match_so_far.matchable_references.size() >= 1);
              assert(best_match_so_far.object_references.size() >= 1);

              // ds replace the best with this match on the spot - we don't have to update the query
              // information
              best_match_so_far.matchable_references =
                std::move(std::vector<const Matchable*>(1, matchable_reference));
              best_match_so_far.object_references =
                std::move(std::vector<ObjectType>(1, object_reference));
              best_match_so_far.distance = distance;
              assert(best_match_so_far.matchable_references.size() == 1);
              assert(best_match_so_far.object_references.size() == 1);
            }

            // ds if the match is equal to the last (multiple candidates)
            else if (distance == best_distance_so_far) {
              // ds add the candidate - we don't have to update the distance since its the same as
              // the best
              best_match_so_far.matchable_references.push_back(matchable_reference);
              best_match_so_far.object_references.push_back(object_reference);
              assert(best_match_so_far.matchable_references.size() > 1);
              assert(best_match_so_far.object_references.size() > 1);
            }
          } catch (const std::out_of_range& /*exception*/) {
            // ds add a new match
            best_matches_.insert(std::make_pair(
              identifer_tree_reference,
              Match(
                matchable_query_, matchable_reference, object_query, object_reference, distance)));
          }
        }
      }
    }
#endif

    //! @brief recursively counts all leafs and descriptors stored in the tree (expensive)
    //! @param[in] starting node (only subtree will be evaluated)
    //! @param[out] number_of_leafs_
    //! @param[out] number_of_matchables_
    //! @param[out] leafs_
    void _getLeafs(Node* node_,
                   uint64_t& number_of_leafs_,
                   uint64_t& number_of_matchables_,
                   std::vector<const Node*>& leafs_) const {
      // ds handle call on empty tree without node_ == _root
      if (_root == nullptr) {
        number_of_leafs_      = 0;
        number_of_matchables_ = 0;
        leafs_.clear();
        return;
      }

      // ds if we are in a node (that has leafs)
      if (node_->has_leafs) {
        assert(node_->right);
        assert(node_->left);

        // ds look deeper on left and right subtree
        _getLeafs(node_->left, number_of_leafs_, number_of_matchables_, leafs_);
        _getLeafs(node_->right, number_of_leafs_, number_of_matchables_, leafs_);

      } else {
        // ds we reached a leaf
        assert(!node_->right);
        assert(!node_->left);

        // ds update statistics and terminate recursion
        ++number_of_leafs_;
        number_of_matchables_ += node_->matchables.size();
        leafs_.push_back(node_);
      }
    }

    // ds public attributes
  public:
#ifdef SRRG_MERGE_DESCRIPTORS
    //! @brief maximum allowed descriptor distance for merging two descriptors
    static uint32_t maximum_distance_for_merge;
#endif

    // ds attributes
  protected:
    //! @brief serializable header carrying core attributes
    mutable Header _header;

    //! @brief root node (e.g. starting point for similarity search)
    Node* _root = nullptr;

    //! @brief bookkeeping: all matchables contained in the tree
    MatchableVector _matchables;
    MatchableVector _matchables_to_train;

    //! @brief bookkeeping: integrated matchable train identifiers (unique)
    std::set<uint64_t> _added_identifiers_train;

    //! @brief bookkeeping: trainable matchables resulting from last matchAndAdd call
    std::vector<Trainable> _trainables;

#ifdef SRRG_MERGE_DESCRIPTORS
    //! @brief bookkeeping: merged matchable pairs (query -> reference) resulting from last
    //! matchAndAdd call over Mergable.query one has access to the merged (=freed) matchable and can
    //! update external bookkeeping accordingly
    MatchableMergeVector _merged_matchables;

    //! statistics
    size_t _number_of_merged_matchables_last_training = 0;
#endif
  };

// ds default configuration
#ifdef SRRG_MERGE_DESCRIPTORS
  template <typename BinaryNodeType_>
  uint32_t BinaryTree<BinaryNodeType_>::maximum_distance_for_merge = 0;
#endif

  template <typename ObjectType_>
  using BinaryTree128 = BinaryTree<BinaryNode128<ObjectType_>>;
  template <typename ObjectType_>
  using BinaryTree256 = BinaryTree<BinaryNode256<ObjectType_>>;
  template <typename ObjectType_>
  using BinaryTree512 = BinaryTree<BinaryNode512<ObjectType_>>;

} // namespace srrg_hbst
