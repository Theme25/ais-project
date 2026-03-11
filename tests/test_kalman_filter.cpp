#include "kalman_filter.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

// ── Minimal test framework ────────────────────────────────────────────────────

static int g_passed = 0;
static int g_failed = 0;

#define ASSERT_TRUE(expr)                                                    \
    do {                                                                     \
        if (!(expr)) {                                                       \
            std::cerr << "  FAIL " << __FILE__ << ":" << __LINE__           \
                      << "  " #expr "\n";                                    \
            ++g_failed;                                                      \
        } else {                                                             \
            ++g_passed;                                                      \
        }                                                                    \
    } while (false)

#define ASSERT_NEAR(a, b, tol)                                               \
    do {                                                                     \
        double _a = (a), _b = (b), _t = (tol);                              \
        if (std::abs(_a - _b) > _t) {                                        \
            std::cerr << "  FAIL " << __FILE__ << ":" << __LINE__           \
                      << "  |" #a " - " #b "| = "                           \
                      << std::abs(_a - _b) << " > " << _t << "\n";         \
            ++g_failed;                                                      \
        } else {                                                             \
            ++g_passed;                                                      \
        }                                                                    \
    } while (false)

#define ASSERT_THROWS(expr, exc_type)                                        \
    do {                                                                     \
        bool _threw = false;                                                 \
        try { (expr); }                                                      \
        catch (const exc_type&) { _threw = true; }                          \
        catch (...) {}                                                       \
        if (!_threw) {                                                       \
            std::cerr << "  FAIL " << __FILE__ << ":" << __LINE__           \
                      << "  expected " #exc_type " from: " #expr "\n";      \
            ++g_failed;                                                      \
        } else {                                                             \
            ++g_passed;                                                      \
        }                                                                    \
    } while (false)

#define RUN_TEST(name)                                                       \
    do {                                                                     \
        std::cout << "[ RUN ] " #name "\n";                                 \
        name();                                                              \
        std::cout << "[  OK ] " #name "\n";                                 \
    } while (false)

// ── Helpers ───────────────────────────────────────────────────────────────────

static double vec3_norm(const std::array<double, 3>& v) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

// After reset the reported position equals the given initial position.
void test_reset_sets_initial_position() {
    DronePoseKalmanFilter kf;
    kf.reset({1.0, 2.0, 3.0});
    const auto pos = kf.position();
    ASSERT_NEAR(pos[0], 1.0, 1e-9);
    ASSERT_NEAR(pos[1], 2.0, 1e-9);
    ASSERT_NEAR(pos[2], 3.0, 1e-9);
}

// After reset the reported velocity equals the given initial velocity.
void test_reset_sets_initial_velocity() {
    DronePoseKalmanFilter kf;
    kf.reset({0.0, 0.0, 0.0}, {1.0, -2.0, 0.5});
    const auto vel = kf.velocity();
    ASSERT_NEAR(vel[0],  1.0, 1e-9);
    ASSERT_NEAR(vel[1], -2.0, 1e-9);
    ASSERT_NEAR(vel[2],  0.5, 1e-9);
}

// Predicting with a nonzero velocity must advance position by v*dt.
void test_predict_advances_position_with_velocity() {
    DronePoseKalmanFilter kf;
    kf.reset({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0});
    kf.predict(0.5);
    const auto pos = kf.position();
    ASSERT_NEAR(pos[0], 0.5, 1e-9);
    ASSERT_NEAR(pos[1], 0.0, 1e-9);
    ASSERT_NEAR(pos[2], 0.0, 1e-9);
}

// predict() must not change the velocity state (constant-velocity model).
void test_predict_does_not_change_velocity() {
    DronePoseKalmanFilter kf;
    kf.reset({0.0, 0.0, 0.0}, {2.0, -1.0, 0.5});
    kf.predict(1.0);
    const auto vel = kf.velocity();
    ASSERT_NEAR(vel[0],  2.0, 1e-9);
    ASSERT_NEAR(vel[1], -1.0, 1e-9);
    ASSERT_NEAR(vel[2],  0.5, 1e-9);
}

// A perfect measurement at the current position should barely move the state.
void test_update_with_exact_measurement_keeps_position() {
    DronePoseKalmanFilter kf;
    kf.reset({1.0, 2.0, 3.0});
    kf.updateWithDronePosition({1.0, 2.0, 3.0}, 0.01);
    const auto pos = kf.position();
    ASSERT_NEAR(pos[0], 1.0, 1e-6);
    ASSERT_NEAR(pos[1], 2.0, 1e-6);
    ASSERT_NEAR(pos[2], 3.0, 1e-6);
}

// After many updates the fused estimate should converge close to the truth.
void test_repeated_updates_converge_to_measurement() {
    DronePoseKalmanFilter kf;
    kf.reset({0.0, 0.0, 0.0});
    const std::array<double, 3> truth = {5.0, -3.0, 2.0};
    for (int i = 0; i < 50; ++i) {
        kf.predict(0.02);
        kf.updateWithDronePosition(truth, 0.01);
    }
    const auto pos = kf.position();
    ASSERT_NEAR(pos[0], truth[0], 0.1);
    ASSERT_NEAR(pos[1], truth[1], 0.1);
    ASSERT_NEAR(pos[2], truth[2], 0.1);
}

// Fusing two sensors should pull the estimate between both readings.
void test_dual_sensor_fusion_interpolates() {
    DronePoseKalmanFilter kf;
    kf.reset({0.0, 0.0, 0.0});
    kf.predict(0.02);

    // Drone says x=1, camera says x=3 with equal variance → expect ~x=2.
    kf.updateWithDronePosition({1.0, 0.0, 0.0}, 0.1);
    kf.updateWithCameraArucoPosition({3.0, 0.0, 0.0}, 0.1);

    const auto pos = kf.position();
    ASSERT_TRUE(pos[0] > 1.0 && pos[0] < 3.0);
}

// Lower variance (more trusted) sensor should dominate the fused estimate.
void test_lower_variance_sensor_dominates() {
    DronePoseKalmanFilter kf;
    kf.reset({0.0, 0.0, 0.0});
    kf.predict(0.02);

    // Drone (very trusted) says x=1; camera (noisy) says x=10.
    kf.updateWithDronePosition({1.0, 0.0, 0.0}, 0.001);
    kf.updateWithCameraArucoPosition({10.0, 0.0, 0.0}, 10.0);

    const auto pos = kf.position();
    // Fused result should sit much closer to drone reading.
    ASSERT_TRUE(pos[0] < 2.0);
}

// compareMeasurements must return the correct residual vector and norm.
void test_compare_measurements_residual() {
    DronePoseKalmanFilter kf;
    kf.reset({0.0, 0.0, 0.0});

    const std::array<double, 3> drone = {1.0, 0.0, 0.0};
    const std::array<double, 3> cam   = {0.0, 0.0, 0.0};
    const auto cmp = kf.compareMeasurements(drone, cam);

    ASSERT_NEAR(cmp.residual[0], 1.0, 1e-9);
    ASSERT_NEAR(cmp.residual[1], 0.0, 1e-9);
    ASSERT_NEAR(cmp.residual[2], 0.0, 1e-9);
    ASSERT_NEAR(cmp.residual_norm, 1.0, 1e-9);
}

// compareMeasurements with matching sensors should give zero residual.
void test_compare_measurements_zero_residual_when_equal() {
    DronePoseKalmanFilter kf;
    kf.reset({0.0, 0.0, 0.0});

    const std::array<double, 3> pos = {2.0, -1.0, 3.0};
    const auto cmp = kf.compareMeasurements(pos, pos);

    ASSERT_NEAR(cmp.residual_norm, 0.0, 1e-9);
}

// predict() with dt <= 0 must throw.
void test_predict_negative_dt_throws() {
    DronePoseKalmanFilter kf;
    kf.reset({0.0, 0.0, 0.0});
    ASSERT_THROWS(kf.predict(-0.01), std::invalid_argument);
    ASSERT_THROWS(kf.predict(0.0),   std::invalid_argument);
}

// updateWithDronePosition() with non-positive variance must throw.
void test_update_zero_variance_throws() {
    DronePoseKalmanFilter kf;
    kf.reset({0.0, 0.0, 0.0});
    ASSERT_THROWS(kf.updateWithDronePosition({0.0, 0.0, 0.0}, 0.0),  std::invalid_argument);
    ASSERT_THROWS(kf.updateWithDronePosition({0.0, 0.0, 0.0}, -1.0), std::invalid_argument);
}

// setProcessNoise() with non-positive values must throw.
void test_set_process_noise_invalid_throws() {
    DronePoseKalmanFilter kf;
    ASSERT_THROWS(kf.setProcessNoise(0.0,  0.1), std::invalid_argument);
    ASSERT_THROWS(kf.setProcessNoise(0.1,  0.0), std::invalid_argument);
    ASSERT_THROWS(kf.setProcessNoise(-1.0, 0.1), std::invalid_argument);
}

// Running many predict+update cycles must not produce NaN or Inf in position.
void test_long_run_numerical_stability() {
    DronePoseKalmanFilter kf;
    kf.reset({0.0, 0.0, 1.5}, {0.1, 0.0, 0.0});

    for (int i = 0; i < 1000; ++i) {
        kf.predict(0.02);
        kf.updateWithDronePosition({0.02 * i, 0.0, 1.5}, 0.05);
        kf.updateWithCameraArucoPosition({0.02 * i + 0.01, 0.0, 1.5}, 0.1);
    }

    const auto pos = kf.position();
    ASSERT_TRUE(std::isfinite(pos[0]));
    ASSERT_TRUE(std::isfinite(pos[1]));
    ASSERT_TRUE(std::isfinite(pos[2]));
}

// After reset the velocity is zero by default.
void test_default_velocity_is_zero_after_reset() {
    DronePoseKalmanFilter kf;
    kf.reset({1.0, 2.0, 3.0});
    const auto vel = kf.velocity();
    ASSERT_NEAR(vel[0], 0.0, 1e-9);
    ASSERT_NEAR(vel[1], 0.0, 1e-9);
    ASSERT_NEAR(vel[2], 0.0, 1e-9);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    RUN_TEST(test_reset_sets_initial_position);
    RUN_TEST(test_reset_sets_initial_velocity);
    RUN_TEST(test_predict_advances_position_with_velocity);
    RUN_TEST(test_predict_does_not_change_velocity);
    RUN_TEST(test_update_with_exact_measurement_keeps_position);
    RUN_TEST(test_repeated_updates_converge_to_measurement);
    RUN_TEST(test_dual_sensor_fusion_interpolates);
    RUN_TEST(test_lower_variance_sensor_dominates);
    RUN_TEST(test_compare_measurements_residual);
    RUN_TEST(test_compare_measurements_zero_residual_when_equal);
    RUN_TEST(test_predict_negative_dt_throws);
    RUN_TEST(test_update_zero_variance_throws);
    RUN_TEST(test_set_process_noise_invalid_throws);
    RUN_TEST(test_long_run_numerical_stability);
    RUN_TEST(test_default_velocity_is_zero_after_reset);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
