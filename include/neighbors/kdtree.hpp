#pragma once
#include "neighbors/search.hpp"
#include "distance.hpp"
#include <memory>
#include <vector>

class KDTree : public NeighborSearch {
public:
    explicit KDTree(const Dataset& X);

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
        int index;
        int axis;
        real_t split;
        Node* left;
        Node* right;
    };

    const Dataset& X_;
    std::vector<int> idx_;
    std::unique_ptr<Node> root_;
    int dim_;

    std::unique_ptr<Node> build(
        int l,
        int r,
        int depth
    );

    void radius_node(
        const Node* node,
        const Point& query,
        real_t eps2,
        std::vector<int>& out
    ) const;
};