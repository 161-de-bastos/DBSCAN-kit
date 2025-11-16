#pragma once
#include <string>

struct ParallelConfig {
    std::string mode = "serial";
    int num_threads = 0;
};

struct DBSCANConfig {
    double eps = 0.5;
    int minPts = 5;
};

struct Config {
    std::string task = "dbscan";
    std::string backend = "brute";

    ParallelConfig parallel;
    DBSCANConfig dbscan;
};

bool load_config(
    const std::string& path,
    Config& cfg
);