#include <iostream>
#include <memory>
#include "common.hpp"
#include "dbscan.hpp"
#include "neighbors/brute.hpp"
#include "neighbors/kdtree.hpp"
#include "neighbors/balltree.hpp"

std::unique_ptr<NeighborSearch> make_backend(
    const Dataset& X,
    const std::string& which
) {
    if (which == "brute") {
        return std::make_unique<bruteForce>(X);
    } else if (which == "kdtree") {
        return std::make_unique<KDTree>(X);
    } else if (which == "balltree") {
        return std::make_unique<BallTree>(X);
    }
    return std::make_unique<bruteForce>(X);
}

int main(int argc, char** argv) {
    Dataset X = {
        {0.0, 0.0},
        {0.1, 0.0},
        {0.0, 0.1},
        {5.0, 5.0},
        {5.1, 5.0},
        {5.0, 5.1},
        {10.0, 10.0}
    };

    std::string backend = "brute";
    if (argc > 1) {
        backend = argv[1];
    }

    DBSCANParams params;
    params.eps     = 0.3;
    params.minPts = 3;

    auto ns = make_backend(X, backend);
    DBSCAN db(X, std::move(ns));
    DBSCANResult res = db.run(params);

    std::cout << "Backend: " << backend << "\n";
    std::cout << "clusters=" << res.n_clusters
              << " noise=" << res.n_noise << "\n";

    for (std::size_t i = 0; i < X.size(); ++i) {
        std::cout << "punto " << i << " -> label=" << res.labels[i] << "\n";
    }
    return 0;
}
