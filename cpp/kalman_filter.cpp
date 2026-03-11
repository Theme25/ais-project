#include "kalman_filter.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

inline double& mat(std::array<double, 36>& m, int r, int c) { return m[r * 6 + c]; }
inline double matc(const std::array<double, 36>& m, int r, int c) { return m[r * 6 + c]; }

void set_identity(std::array<double, 36>& m, double scale = 1.0) {
    std::fill(m.begin(), m.end(), 0.0);
    for (int i = 0; i < 6; ++i) {
        mat(m, i, i) = scale;
    }
}

}  // namespace

DronePoseKalmanFilter::DronePoseKalmanFilter() {
    set_identity(P_, 1.0);
}

void DronePoseKalmanFilter::setProcessNoise(double position_noise, double velocity_noise) {
    if (position_noise <= 0.0 || velocity_noise <= 0.0) {
        throw std::invalid_argument("process noise must be > 0");
    }

    process_position_noise_ = position_noise;
    process_velocity_noise_ = velocity_noise;
}

void DronePoseKalmanFilter::reset(const std::array<double, 3>& position,
                                  const std::array<double, 3>& velocity) {
    x_ = {position[0], position[1], position[2], velocity[0], velocity[1], velocity[2]};
    set_identity(P_, 1.0);
}

void DronePoseKalmanFilter::predict(double dt_seconds) {
    if (dt_seconds <= 0.0) {
        throw std::invalid_argument("dt_seconds must be > 0");
    }

    // x_k = F * x_{k-1}
    x_[0] += dt_seconds * x_[3];
    x_[1] += dt_seconds * x_[4];
    x_[2] += dt_seconds * x_[5];

    // Build F
    std::array<double, 36> F{};
    set_identity(F, 1.0);
    mat(F, 0, 3) = dt_seconds;
    mat(F, 1, 4) = dt_seconds;
    mat(F, 2, 5) = dt_seconds;

    // A = F * P
    std::array<double, 36> A{};
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            double sum = 0.0;
            for (int k = 0; k < 6; ++k) {
                sum += matc(F, r, k) * matc(P_, k, c);
            }
            mat(A, r, c) = sum;
        }
    }

    // P = A * F^T
    std::array<double, 36> nextP{};
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            double sum = 0.0;
            for (int k = 0; k < 6; ++k) {
                sum += matc(A, r, k) * matc(F, c, k);
            }
            mat(nextP, r, c) = sum;
        }
    }

    // Constant-velocity process noise model (discrete time), with
    // position/velocity cross-correlation terms.
    //
    // For each axis [p, v], we add:
    // [ q_p*dt^3/3      q_p*dt^2/2 ]
    // [ q_p*dt^2/2      q_v*dt     ]
    //
    // This avoids the filter becoming overconfident in velocity.
    const double dt2 = dt_seconds * dt_seconds;
    const double dt3 = dt2 * dt_seconds;
    for (int i = 0; i < 3; ++i) {
        mat(nextP, i,     i)     += process_position_noise_ * dt3 / 3.0;
        mat(nextP, i + 3, i + 3) += process_velocity_noise_ * dt_seconds;
        mat(nextP, i,     i + 3) += process_position_noise_ * dt2 / 2.0;
        mat(nextP, i + 3, i)     += process_position_noise_ * dt2 / 2.0;
    }

    // Enforce symmetry to reduce numerical drift over long runs.
    for (int r = 0; r < 6; ++r) {
        for (int c = r + 1; c < 6; ++c) {
            const double avg = 0.5 * (matc(nextP, r, c) + matc(nextP, c, r));
            mat(nextP, r, c) = avg;
            mat(nextP, c, r) = avg;
        }
    }

    P_ = nextP;
}

void DronePoseKalmanFilter::updateWithDronePosition(const std::array<double, 3>& position,
                                                    double position_variance) {
    // Intentionally shares the same position-only measurement model as camera updates.
    // Sensor trust is configured by the caller via `position_variance`.
    updatePositionMeasurement(position, position_variance);
}

void DronePoseKalmanFilter::updateWithCameraArucoPosition(const std::array<double, 3>& position,
                                                          double position_variance) {
    // Intentionally shares the same position-only measurement model as drone updates.
    // Sensor trust is configured by the caller via `position_variance`.
    updatePositionMeasurement(position, position_variance);
}

DronePoseKalmanFilter::ComparisonResult DronePoseKalmanFilter::compareMeasurements(
    const std::array<double, 3>& drone_position,
    const std::array<double, 3>& camera_position) const {
    ComparisonResult result;
    const double dx = drone_position[0] - camera_position[0];
    const double dy = drone_position[1] - camera_position[1];
    const double dz = drone_position[2] - camera_position[2];
    result.residual = {dx, dy, dz};
    result.residual_norm = std::sqrt(dx * dx + dy * dy + dz * dz);
    return result;
}

std::array<double, 3> DronePoseKalmanFilter::position() const {
    return {x_[0], x_[1], x_[2]};
}

std::array<double, 3> DronePoseKalmanFilter::velocity() const {
    return {x_[3], x_[4], x_[5]};
}

void DronePoseKalmanFilter::updatePositionMeasurement(const std::array<double, 3>& z,
                                                      double measurement_variance) {
    if (measurement_variance <= 0.0) {
        throw std::invalid_argument("measurement_variance must be > 0");
    }

    // Innovation y = z - Hx, where H selects position states.
    double y[3] = {z[0] - x_[0], z[1] - x_[1], z[2] - x_[2]};

    // S = HPH^T + R => top-left 3x3 of P + variance*I.
    double S[3][3]{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            S[r][c] = matc(P_, r, c);
            if (r == c) {
                S[r][c] += measurement_variance;
            }
        }
    }

    // Inverse of 3x3 matrix S.
    const double det =
        S[0][0] * (S[1][1] * S[2][2] - S[1][2] * S[2][1]) -
        S[0][1] * (S[1][0] * S[2][2] - S[1][2] * S[2][0]) +
        S[0][2] * (S[1][0] * S[2][1] - S[1][1] * S[2][0]);

    if (std::abs(det) < 1e-12) {
        throw std::runtime_error("innovation covariance is singular");
    }

    const double inv_det = 1.0 / det;
    double Sinv[3][3]{};
    Sinv[0][0] = (S[1][1] * S[2][2] - S[1][2] * S[2][1]) * inv_det;
    Sinv[0][1] = (S[0][2] * S[2][1] - S[0][1] * S[2][2]) * inv_det;
    Sinv[0][2] = (S[0][1] * S[1][2] - S[0][2] * S[1][1]) * inv_det;
    Sinv[1][0] = (S[1][2] * S[2][0] - S[1][0] * S[2][2]) * inv_det;
    Sinv[1][1] = (S[0][0] * S[2][2] - S[0][2] * S[2][0]) * inv_det;
    Sinv[1][2] = (S[0][2] * S[1][0] - S[0][0] * S[1][2]) * inv_det;
    Sinv[2][0] = (S[1][0] * S[2][1] - S[1][1] * S[2][0]) * inv_det;
    Sinv[2][1] = (S[0][1] * S[2][0] - S[0][0] * S[2][1]) * inv_det;
    Sinv[2][2] = (S[0][0] * S[1][1] - S[0][1] * S[1][0]) * inv_det;

    // K = P H^T S^-1 => first 3 columns of H^T pick P[:,0:3].
    double K[6][3]{};
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 3; ++c) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += matc(P_, r, k) * Sinv[k][c];
            }
            K[r][c] = sum;
        }
    }

    // x = x + K*y
    for (int r = 0; r < 6; ++r) {
        double dy = 0.0;
        for (int k = 0; k < 3; ++k) {
            dy += K[r][k] * y[k];
        }
        x_[r] += dy;
    }

    // Joseph-ish simplified covariance update: P = (I - K H) P
    std::array<double, 36> IminusKH{};
    set_identity(IminusKH, 1.0);
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 3; ++c) {
            mat(IminusKH, r, c) -= K[r][c];
        }
    }

    std::array<double, 36> nextP{};
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            double sum = 0.0;
            for (int k = 0; k < 6; ++k) {
                sum += matc(IminusKH, r, k) * matc(P_, k, c);
            }
            mat(nextP, r, c) = sum;
        }
    }

    P_ = nextP;
}
