#pragma once
#include "neighbors/search.hpp"
#include "distance.hpp"

class bruteForce : public NeighborSearch {
public:
    explicit bruteForce(const Dataset& X);

    void radius_query(
        int idx, 
        real_t eps,
        std::vector<int>& out_idx
    ) const override;

    int size() const override;

private:
    const Dataset& X_;
};