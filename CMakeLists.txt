cmake_minimum_required(VERSION 3.14)
project(drone_pose_kalman LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(kalman_filter STATIC kalman_filter.cpp)
target_include_directories(kalman_filter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(kalman_example kalman_example.cpp)
target_link_libraries(kalman_example PRIVATE kalman_filter)

enable_testing()

add_executable(test_kalman_filter test_kalman_filter.cpp)
target_link_libraries(test_kalman_filter PRIVATE kalman_filter)

add_test(NAME kalman_filter_tests COMMAND test_kalman_filter)
