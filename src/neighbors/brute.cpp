#include "neighbors/brute.hpp"

bruteForce::bruteForce(const Dataset& X) : X_(X) {}

int bruteForce::size() const {
    return static_cast<int>(X_.size());
}

void bruteForce::radius_query(
    int idx, 
    real_t eps,
    std::vector<int>& out_idx
) const {
    out_idx.clear();
    const int n = static_cast<int>(X_.size());
    const Point& p = X_[idx];
    const real_t eps2 = eps * eps;
    
    for (int j = 0; j < n; ++j) {
        if (j == idx) continue;
        if (euclidean_dist(p,X_[j]) <= eps2) out_idx.push_back(j);
    }
}
