import unittest
from unittest.mock import patch

import numpy as np

from detect_goal import detectGoal
from transform_tree import initTree, make_transform


MATLAB_CAMERA_POS = np.array([0.1, 0.0, 0.0])
MATLAB_CAMERA_OR = np.array([np.pi / 2.0, np.pi, 0.0])
MATLAB_CAMERA_MATRIX = np.array(
    [
        [603.816409588989, 0.0, 388.676982266589],
        [0.0, 600.17202631038163, 240.555861086624],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
MATLAB_DIST_COEFFS = np.array([[0.043102806701415232, -0.1082637934138599, 0.0, 0.0, 0.0]], dtype=np.float64)


class _FakeDetector:
    def __init__(self, corners, ids):
        self._corners = corners
        self._ids = ids

    def detectMarkers(self, _img):
        return self._corners, self._ids, None


class DetectGoalTests(unittest.TestCase):
    def test_no_detection_sentinel_outputs(self):
        tree = initTree(MATLAB_CAMERA_POS, MATLAB_CAMERA_OR, np.zeros(3), np.zeros(3))

        with patch("detect_goal._aruco_detector", return_value=_FakeDetector([], None)):
            ids, aruco_pos = detectGoal(
                img=np.zeros((10, 10, 3), dtype=np.uint8),
                tree=tree,
                posUAV=np.zeros(3),
                orUAV=np.zeros(3),
                cameraMatrix=MATLAB_CAMERA_MATRIX,
                distCoeffs=MATLAB_DIST_COEFFS,
            )

        self.assertEqual(ids.shape, (1, 1))
        self.assertEqual(aruco_pos.shape, (3, 1))
        np.testing.assert_allclose(ids, [[0.0]])
        np.testing.assert_allclose(aruco_pos, np.zeros((3, 1)))

    def test_marker_101_updates_goal_path(self):
        tree = initTree(MATLAB_CAMERA_POS, MATLAB_CAMERA_OR, np.zeros(3), np.zeros(3))

        corners = [np.zeros((1, 4, 2), dtype=np.float32)]
        ids = np.array([[101]], dtype=np.int32)
        rvec = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
        tvec = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)

        with patch("detect_goal._aruco_detector", return_value=_FakeDetector(corners, ids)), patch(
            "detect_goal.cv2.solvePnP", return_value=(True, rvec, tvec)
        ), patch("detect_goal.cv2.Rodrigues", return_value=(np.eye(3), None)):
            out_ids, out_pos = detectGoal(
                img=np.zeros((10, 10, 3), dtype=np.uint8),
                tree=tree,
                posUAV=np.zeros(3),
                orUAV=np.zeros(3),
                cameraMatrix=MATLAB_CAMERA_MATRIX,
                distCoeffs=MATLAB_DIST_COEFFS,
            )

        np.testing.assert_allclose(out_ids, [[101.0]])
        expected_tf = (
            make_transform(np.zeros(3), np.zeros(3))
            @ make_transform(MATLAB_CAMERA_POS, MATLAB_CAMERA_OR)
            @ make_transform(np.array([1.0, 2.0, 3.0]), np.zeros(3))
        )
        np.testing.assert_allclose(out_pos[:, 0], expected_tf[:3, 3])
        self.assertEqual(tree.pathToGoal.ids[2], "101")

    def test_marker_position_uses_nontrivial_uav_cam_chain(self):
        tree = initTree(
            posCam=np.array([1.0, 0.0, 0.0]),
            orCam=np.zeros(3),
            posUAV=np.zeros(3),
            orUAV=np.array([np.pi / 2, 0.0, 0.0]),
        )

        corners = [np.zeros((1, 4, 2), dtype=np.float32)]
        ids = np.array([[101]], dtype=np.int32)
        rvec = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
        tvec = np.array([[1.0], [0.0], [0.0]], dtype=np.float64)

        with patch("detect_goal._aruco_detector", return_value=_FakeDetector(corners, ids)), patch(
            "detect_goal.cv2.solvePnP", return_value=(True, rvec, tvec)
        ), patch("detect_goal.cv2.Rodrigues", return_value=(np.eye(3), None)):
            _, out_pos = detectGoal(
                img=np.zeros((10, 10, 3), dtype=np.uint8),
                tree=tree,
                posUAV=np.zeros(3),
                orUAV=np.array([np.pi / 2, 0.0, 0.0]),
                cameraMatrix=np.eye(3),
                distCoeffs=np.zeros(5),
            )

        np.testing.assert_allclose(out_pos[:, 0], [0.0, 2.0, 0.0], atol=1e-9)

    def test_unknown_marker_warns_and_returns_zero_position(self):
        tree = initTree(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))

        corners = [np.zeros((1, 4, 2), dtype=np.float32)]
        ids = np.array([[999]], dtype=np.int32)
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)

        with patch("detect_goal._aruco_detector", return_value=_FakeDetector(corners, ids)), patch(
            "detect_goal.cv2.solvePnP", return_value=(True, rvec, tvec)
        ), patch("detect_goal.cv2.Rodrigues", return_value=(np.eye(3), None)), self.assertWarnsRegex(
            RuntimeWarning, "Unknown marker id 999"
        ):
            out_ids, out_pos = detectGoal(
                img=np.zeros((10, 10, 3), dtype=np.uint8),
                tree=tree,
                posUAV=np.zeros(3),
                orUAV=np.zeros(3),
                cameraMatrix=np.eye(3),
                distCoeffs=np.zeros(5),
            )

        np.testing.assert_allclose(out_ids, [[999.0]])
        np.testing.assert_allclose(out_pos[:, 0], [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
