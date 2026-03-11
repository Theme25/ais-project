#pragma once

#include <array>

class DronePoseKalmanFilter {
public:
    struct ComparisonResult {
        std::array<double, 3> residual{};  // drone - camera
        double residual_norm{0.0};
    };

    DronePoseKalmanFilter();

    void setProcessNoise(double position_noise, double velocity_noise);
    void reset(const std::array<double, 3>& position,
               const std::array<double, 3>& velocity = {0.0, 0.0, 0.0});

    void predict(double dt_seconds);

    void updateWithDronePosition(const std::array<double, 3>& position,
                                 double position_variance);
    void updateWithCameraArucoPosition(const std::array<double, 3>& position,
                                       double position_variance);

    ComparisonResult compareMeasurements(const std::array<double, 3>& drone_position,
                                         const std::array<double, 3>& camera_position) const;

    std::array<double, 3> position() const;
    std::array<double, 3> velocity() const;

private:
    void updatePositionMeasurement(const std::array<double, 3>& z,
                                   double measurement_variance);

    // x = [px, py, pz, vx, vy, vz]
    std::array<double, 6> x_{};

    // P covariance (6x6)
    std::array<double, 36> P_{};

    double process_position_noise_{0.05};
    double process_velocity_noise_{0.2};
};
