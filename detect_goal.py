"""MATLAB-compatible Python replacement for detectGoal (OpenCV 4.11).

This file focuses on one responsibility:
- read ArUco detections from an image
- update the transform tree the same way MATLAB-generated code does
- output MATLAB-shaped arrays so downstream code can keep existing assumptions
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Any

import cv2
import numpy as np

# Tree helpers are kept in a separate module so transform logic stays reusable.
from transform_tree import apply_chain, path_for_marker, updateTreeAfter, update_uav_transform


# Project requirement: standard ArUco marker is 15 cm.
MARKER_SIZE_M = 0.15


def _marker_object_points(marker_size_m: float = MARKER_SIZE_M) -> np.ndarray:
    """Return 3D corner points for a square marker centered at origin."""
    half = marker_size_m / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


@dataclass
class DetectGoalResult:
    """Python-friendly wrapper around the MATLAB-style outputs.

    ids: MATLAB orientation (1, N)
    arucoPos: MATLAB orientation (3, N)
    """

    ids: np.ndarray
    arucoPos: np.ndarray


def rotm2eul_zyx(r: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to ZYX Euler angles (yaw, pitch, roll).

    This mirrors MATLAB's ZYX convention used by the generated code.
    """
    # Normal case (no gimbal lock).
    if abs(r[2, 0]) < 1.0:
        y = np.arcsin(-r[2, 0])
        z = np.arctan2(r[1, 0], r[0, 0])
        x = np.arctan2(r[2, 1], r[2, 2])
    else:
        # Gimbal-lock fallback branch.
        y = np.pi / 2 if r[2, 0] <= -1 else -np.pi / 2
        z = np.arctan2(-r[0, 1], r[1, 1])
        x = 0.0

    return np.array([z, y, x], dtype=np.float64)


def _aruco_detector(aruco_dict: int) -> cv2.aruco.ArucoDetector:
    """Create detector object for a selected predefined dictionary."""
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, params)


def detectGoal(
    img: np.ndarray,
    tree: Any,
    posUAV: np.ndarray,
    orUAV: np.ndarray,
    cameraMatrix: np.ndarray,
    distCoeffs: np.ndarray,
    arucoDict: int = cv2.aruco.DICT_4X4_50,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop-in replacement for MATLAB-generated detectGoal.

    Steps:
    1) Update UAV pose in both pathToGoal and pathToStart.
    2) Detect markers in current frame.
    3) Estimate marker pose (rvec/tvec) using calibration.
    4) Update tree with marker transform.
    5) Apply transform chain and export world/local position.

    Returns
    -------
    ids: np.ndarray
        Shape (1, N). If no detection: [[0.0]].
    arucoPos: np.ndarray
        Shape (3, N). If no detection: zeros((3,1)).
    """

    # Support both object-style and dict-style tree containers.
    path_to_goal = tree.pathToGoal if hasattr(tree, "pathToGoal") else tree["pathToGoal"]
    path_to_start = tree.pathToStart if hasattr(tree, "pathToStart") else tree["pathToStart"]

    # Keep UAV node transform current for both graph branches.
    update_uav_transform(path_to_goal, np.asarray(posUAV), np.asarray(orUAV))
    update_uav_transform(path_to_start, np.asarray(posUAV), np.asarray(orUAV))

    detector = _aruco_detector(arucoDict)
    corners, ids, _ = detector.detectMarkers(img)

    # MATLAB-style "no detection" sentinel behavior.
    if ids is None or len(ids) == 0:
        return np.array([[0.0]], dtype=np.float64), np.zeros((3, 1), dtype=np.float64)

    # MATLAB output shape for IDs: one row, N columns.
    ids_flat = ids.flatten().astype(np.int32)
    ids_out = ids_flat.reshape(1, -1).astype(np.float64)

    object_points = _marker_object_points(MARKER_SIZE_M)

    # MATLAB output shape: 3 x N.
    aruco_pos = np.zeros((3, len(ids_flat)), dtype=np.float64)

    for i, marker_id in enumerate(ids_flat):
        image_points = np.asarray(corners[i][0], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            cameraMatrix,
            distCoeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            warnings.warn(f"solvePnP failed for marker {int(marker_id)}", RuntimeWarning)
            continue

        # Rotation and translation of marker in camera frame.
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)

        # Convert OpenCV Rodrigues vector to rotation matrix then to ZYX Euler.
        rotm, _ = cv2.Rodrigues(rvec)
        or_arc = rotm2eul_zyx(rotm)
        pos_arc = tvec

        # Route marker ID to correct path (101=>goal, 102=>start).
        path = path_for_marker(tree, int(marker_id))

        if path is None:
            # Unknown ID: keep identity (same behavior as default branch in reference).
            warnings.warn(
                f"Unknown marker id {int(marker_id)} (supported: 101=>goal, 102=>start)",
                RuntimeWarning,
            )
            transform = np.eye(4, dtype=np.float64)
        else:
            # Update marker node then compose full transform chain.
            updateTreeAfter(path, str(marker_id), pos_arc, or_arc)
            transform = apply_chain(path)

        # Final position is translation component of 4x4 transform.
        aruco_pos[:, i] = transform[:3, 3]

    return ids_out, aruco_pos


def detect_goal(*args: Any, **kwargs: Any) -> DetectGoalResult:
    """Snake_case convenience wrapper for Python-native callers."""
    ids, aruco_pos = detectGoal(*args, **kwargs)
    return DetectGoalResult(ids=ids, arucoPos=aruco_pos)
