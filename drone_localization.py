"""Aruco-based drone localization with OpenCV + Kalman filter demo.

This script is a standalone development/demo utility.
Production expectation from the project is to run Kalman in C++.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np


# Supported predefined dictionaries for quick switching from CLI.
ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
}

# MATLAB cameraIntrinsics constants provided by user snippet.
MATLAB_FOCAL_LENGTH = (603.816409588989, 600.17202631038163)
MATLAB_PRINCIPAL_POINT = (388.676982266589, 240.555861086624)
MATLAB_RADIAL = (0.043102806701415232, -0.1082637934138599)
MATLAB_TANGENTIAL = (0.0, 0.0)
MATLAB_IMAGE_SIZE = (480.0, 768.0)  # [rows, cols]


@dataclass
class PoseEstimate:
    """Container for one detected marker pose."""

    marker_id: int
    rvec: np.ndarray
    tvec: np.ndarray


def build_kalman_filter(dt: float = 1.0 / 30.0) -> cv2.KalmanFilter:
    """Create constant-velocity Kalman filter over 3D position.

    State: [x, y, z, vx, vy, vz]
    Meas : [x, y, z]
    """
    kf = cv2.KalmanFilter(6, 3)

    # Motion model (position integrates velocity each frame).
    kf.transitionMatrix = np.array(
        [
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Measurement model: we observe only position directly.
    kf.measurementMatrix = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    # Default noise settings; tune for your drone/camera behavior.
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 5e-2

    # Initial posterior covariance/state.
    kf.errorCovPost = np.eye(6, dtype=np.float32)
    kf.statePost = np.zeros((6, 1), dtype=np.float32)
    return kf


def matlab_intrinsics_to_opencv() -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Convert MATLAB cameraIntrinsics fields into OpenCV arrays."""
    fx, fy = MATLAB_FOCAL_LENGTH
    cx, cy = MATLAB_PRINCIPAL_POINT
    k1, k2 = MATLAB_RADIAL
    p1, p2 = MATLAB_TANGENTIAL

    # OpenCV camera matrix format.
    camera_matrix = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # OpenCV distortion order: [k1, k2, p1, p2, k3].
    # MATLAB snippet contains only 2 radial terms -> set k3 to 0.
    dist_coeffs = np.array([[k1, k2, p1, p2, 0.0]], dtype=np.float64)

    # Height/width expected by calibration.
    image_size_hw = (int(MATLAB_IMAGE_SIZE[0]), int(MATLAB_IMAGE_SIZE[1]))
    return camera_matrix, dist_coeffs, image_size_hw


def load_calibration(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, tuple[int, int] | None]:
    """Load calibration from either MATLAB constants or NPY files."""
    if args.use_matlab_intrinsics:
        return matlab_intrinsics_to_opencv()

    if not args.camera_matrix or not args.dist_coeffs:
        raise ValueError(
            "Provide both --camera-matrix and --dist-coeffs, or pass --use-matlab-intrinsics"
        )

    camera_matrix = np.load(args.camera_matrix)
    dist_coeffs = np.load(args.dist_coeffs)

    # Normalize 1D distortion arrays to shape (1, N).
    if dist_coeffs.ndim == 1:
        dist_coeffs = dist_coeffs.reshape(1, -1)

    return camera_matrix, dist_coeffs, None


def detect_pose(
    frame: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_size_m: float,
) -> PoseEstimate | None:
    """Detect first ArUco marker and estimate its pose with solvePnP."""
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None or len(ids) == 0:
        return None

    # Draw marker borders and IDs for visualization.
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    half = marker_size_m / 2.0
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )

    # Demo picks the first marker only.
    idx = 0
    marker_id = int(ids[idx][0])
    image_points = np.asarray(corners[idx][0], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None

    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)

    # Draw local axes on top of marker for debugging orientation.
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_size_m * 0.5)
    return PoseEstimate(marker_id=marker_id, rvec=rvec, tvec=tvec)


def annotate(frame: np.ndarray, raw_pos: np.ndarray | None, filt_pos: np.ndarray) -> None:
    """Overlay raw and filtered position text on image."""
    y = 30

    if raw_pos is not None:
        cv2.putText(
            frame,
            f"raw xyz(m): [{raw_pos[0]: .2f}, {raw_pos[1]: .2f}, {raw_pos[2]: .2f}]",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        y += 25

    cv2.putText(
        frame,
        f"kf  xyz(m): [{filt_pos[0]: .2f}, {filt_pos[1]: .2f}, {filt_pos[2]: .2f}]",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--marker-size", type=float, required=True, help="Marker size in meters")
    parser.add_argument("--dict", default="DICT_4X4_50", choices=sorted(ARUCO_DICT_MAP))
    parser.add_argument("--camera-matrix", help="Path to camera_matrix .npy")
    parser.add_argument("--dist-coeffs", help="Path to dist_coeffs .npy")
    parser.add_argument(
        "--use-matlab-intrinsics",
        action="store_true",
        help="Use the MATLAB cameraIntrinsics values provided by the user",
    )
    return parser.parse_args()


def main() -> None:
    """Run real-time loop: capture -> detect -> filter -> display."""
    args = parse_args()
    camera_matrix, dist_coeffs, calibration_image_size = load_calibration(args)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[args.dict])
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera_id}")

    kf = build_kalman_filter()
    last_t = time.time()
    warned_size_mismatch = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Warn once if live stream resolution doesn't match calibration resolution.
            if calibration_image_size is not None and not warned_size_mismatch:
                h, w = frame.shape[:2]
                if (h, w) != calibration_image_size:
                    print(
                        "WARNING: camera resolution does not match calibration image size "
                        f"(live={(h, w)}, calib={calibration_image_size})."
                    )
                    warned_size_mismatch = True

            # Per-frame delta time for dynamic transition update.
            now = time.time()
            dt = max(now - last_t, 1e-3)
            last_t = now

            # Update position<-velocity coefficients using real dt.
            kf.transitionMatrix[0, 3] = dt
            kf.transitionMatrix[1, 4] = dt
            kf.transitionMatrix[2, 5] = dt

            # Prediction always runs even when no marker is visible.
            pred = kf.predict()
            filtered = pred[:3].reshape(3)
            raw_pos = None

            # If marker is observed, use measurement update.
            pose = detect_pose(frame, detector, camera_matrix, dist_coeffs, args.marker_size)
            if pose is not None:
                raw_pos = pose.tvec.astype(np.float32)
                corrected = kf.correct(raw_pos.reshape(3, 1))
                filtered = corrected[:3].reshape(3)

                # Show detected marker ID.
                cv2.putText(
                    frame,
                    f"id: {pose.marker_id}",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            annotate(frame, raw_pos, filtered)
            cv2.imshow("Aruco + Kalman", frame)

            # Quit loop on q key.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Always release resources.
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
