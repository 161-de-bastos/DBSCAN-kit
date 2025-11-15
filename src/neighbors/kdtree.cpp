#include "neighbors/kdtree.hpp"
#include <algorithm>
#include <stdexcept>

KDTree::KDTree(const Dataset& X)
: X_(X) {
    int n = static_cast<int>(X_.size());
    dim_ = (n > 0) ? static_cast<int>(X_[0].size()) : 0;
    idx_.resize(n);
    for (int i = 0; i < n; ++i) idx_[i] = i;
    if (n > 0 && dim_ > 0) root_ = build(0, n, 0);
}

std::unique_ptr<KDTree::Node>
KDTree::build(
    int l,
    int r,
    int depth
){
    if (l >= r) return nullptr;
    int axis = depth % dim_;
    int m = (l + r) / 2;

    std::nth_element(
        idx_.begin() + l,
        idx_.begin() + m,
        idx_.begin() + r,
        [&](int a, int b){
            return X_[a][axis] < X_[b][axis];
        }
    );

    auto node = std::make_unique<Node>();
    node->index = idx_[m];
    node->axis = axis;
    node->split = X_[node->index][axis];
    node->left = build(l, m, depth + 1).release();
    node->right = build(m + 1, r, depth + 1).release();
    return node;
}

void KDTree::radius_node(
    const Node* node,
    const Point& query,
    real_t eps2,
    std::vector<int>& out
) const {
    if (!node) return;

    const Point& p = X_[node->index];
    real_t dist = euclidean_dist(query, p);
    if (dist <= eps2) out.push_back(node->index);

    int axis = node->axis;
    real_t diff = query[axis] - node->split;
    real_t diff2 = diff * diff;

    Node* near = (diff <= 0.0) ? node->left : node->right;
    Node* far = (diff <= 0.0) ? node->right : node->left;

    if (near) radius_node(near, query, eps2, out);
    if (far && diff2 <= eps2) radius_node(far, query, eps2, out);
}

void KDTree::radius_query(
    int idx,
    real_t eps,
    std::vector<int>& out
) const {
    out.clear();
    if (!root_) return;
    const Point& query = X_[idx];
    real_t eps2 = eps * eps;
    radius_node(root_.get(), query, eps2, out);

    out.erase(
        std::remove(
            out.begin(),
            out.end(),
            idx
        ),
        out.end()
    );
}