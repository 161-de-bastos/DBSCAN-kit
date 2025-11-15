#pragma once
#include "neighbors/search.hpp"
#include "distance.hpp"
#include <memory>
#include <vector>

class BallTree : NeighborSearch{
public:
    explicit BallTree(const Dataset& X);

    void radius_query(
        int idx, 
        real_t eps,
        std::vector<int>& out_idx
    ) const override;

    int size() const override {
        return static_cast<int>(X_.size());
    }

private:
    struct Node {
        std::vector<int> indexi;
        Point center;
        real_t radius;
        Node* left;
        Node* right;
    };

    const Dataset& X_;
    std::unique_ptr<Node> root_;
    int dim_;
    int leaf_size_ = 16;

    std::unique_ptr<Node> build(const std::vector<int>& idxs);
    void compute_center(Node* node);
    
    void radius_node(
        const Node* node,
        const Point& query,
        real_t eps2,
        std::vector<int>& out
    ) const;
};