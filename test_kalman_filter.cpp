#include "kalman_filter.hpp"

#include <array>
#include <cmath>
#include <iostream>

namespace {

bool near(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) <= tol;
}

}  // namespace

int main() {
    DronePoseKalmanFilter kf;
    kf.reset({0.0, 0.0, 0.0});

    // Compare helper should match Euclidean residual.
    {
        const std::array<double, 3> drone = {1.0, 2.0, 3.0};
        const std::array<double, 3> cam = {0.0, 0.0, 0.0};
        const auto cmp = kf.compareMeasurements(drone, cam);
        if (!near(cmp.residual[0], 1.0) || !near(cmp.residual[1], 2.0) ||
            !near(cmp.residual[2], 3.0) || !near(cmp.residual_norm, std::sqrt(14.0))) {
            std::cerr << "compareMeasurements failed\n";
            return 1;
        }
    }

    // Predict + update should move estimate toward measurement.
    kf.predict(0.02);
    kf.updateWithDronePosition({1.0, 0.0, 0.0}, 0.01);
    const auto pos = kf.position();
    if (pos[0] <= 0.0) {
        std::cerr << "update did not move state toward measurement\n";
        return 1;
    }

    return 0;
}
