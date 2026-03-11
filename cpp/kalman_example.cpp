#include "kalman_filter.hpp"

#include <array>
#include <iostream>

int main() {
    DronePoseKalmanFilter kf;
    kf.reset({0.0, 0.0, 0.0});

    // Fake timeline (50 Hz) with two independent sensors for position.
    constexpr double dt = 0.02;
    for (int i = 0; i < 5; ++i) {
        kf.predict(dt);

        std::array<double, 3> drone_pos = {0.1 * i, 0.0, 1.0};
        std::array<double, 3> cam_pos = {0.1 * i + 0.02, -0.01, 0.99};

        auto cmp = kf.compareMeasurements(drone_pos, cam_pos);
        std::cout << "measurement residual norm: " << cmp.residual_norm << "\n";

        // Variances are in meters^2.
        // In production, tune these from measured sensor error statistics.
        // Here, drone telemetry is trusted more than camera ArUco estimates.
        kf.updateWithDronePosition(drone_pos, 0.01);
        kf.updateWithCameraArucoPosition(cam_pos, 0.04);

        const auto fused = kf.position();
        std::cout << "fused position: [" << fused[0] << ", " << fused[1] << ", " << fused[2]
                  << "]\n";
    }

    return 0;
}
