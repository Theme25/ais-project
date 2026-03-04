import unittest
from unittest.mock import patch

import numpy as np

from detect_goal import detectGoal
from transform_tree import initTree


class _FakeDetector:
    def __init__(self, corners, ids):
        self._corners = corners
        self._ids = ids

    def detectMarkers(self, _img):
        return self._corners, self._ids, None


class DetectGoalTests(unittest.TestCase):
    def test_no_detection_sentinel_outputs(self):
        tree = initTree(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))

        with patch("detect_goal._aruco_detector", return_value=_FakeDetector([], None)):
            ids, aruco_pos = detectGoal(
                img=np.zeros((10, 10, 3), dtype=np.uint8),
                tree=tree,
                posUAV=np.zeros(3),
                orUAV=np.zeros(3),
                cameraMatrix=np.eye(3),
                distCoeffs=np.zeros(5),
            )

        self.assertEqual(ids.shape, (1, 1))
        self.assertEqual(aruco_pos.shape, (3, 1))
        np.testing.assert_allclose(ids, [[0.0]])
        np.testing.assert_allclose(aruco_pos, np.zeros((3, 1)))

    def test_marker_101_updates_goal_path(self):
        tree = initTree(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))

        corners = [np.zeros((1, 4, 2), dtype=np.float32)]
        ids = np.array([[101]], dtype=np.int32)
        rvecs = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
        tvecs = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float64)

        with patch("detect_goal._aruco_detector", return_value=_FakeDetector(corners, ids)), patch(
            "detect_goal.cv2.aruco.estimatePoseSingleMarkers", return_value=(rvecs, tvecs, None)
        ), patch("detect_goal.cv2.Rodrigues", return_value=(np.eye(3), None)):
            out_ids, out_pos = detectGoal(
                img=np.zeros((10, 10, 3), dtype=np.uint8),
                tree=tree,
                posUAV=np.zeros(3),
                orUAV=np.zeros(3),
                cameraMatrix=np.eye(3),
                distCoeffs=np.zeros(5),
            )

        np.testing.assert_allclose(out_ids, [[101.0]])
        np.testing.assert_allclose(out_pos[:, 0], [1.0, 2.0, 3.0])
        self.assertEqual(tree.pathToGoal.ids[2], "101")


if __name__ == "__main__":
    unittest.main()
