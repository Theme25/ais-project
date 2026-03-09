import unittest

import numpy as np

from transform_tree import apply_chain, initTree, make_transform, updateTreeAfter


class TransformTreeTests(unittest.TestCase):
    def test_init_tree_layout(self):
        tree = initTree(
            posCam=np.array([1.0, 2.0, 3.0]),
            orCam=np.zeros(3),
            posUAV=np.array([4.0, 5.0, 6.0]),
            orUAV=np.zeros(3),
        )

        self.assertEqual(tree.pathToGoal.ids, ["UAV", "CAM", "000"])
        self.assertEqual(tree.pathToStart.ids, ["UAV", "CAM", "000"])
        np.testing.assert_allclose(tree.pathToGoal.tfs[0][:3, 3], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(tree.pathToGoal.tfs[1][:3, 3], [1.0, 2.0, 3.0])

    def test_update_tree_after_updates_cam_child(self):
        tree = initTree(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))

        updateTreeAfter(
            tree.pathToGoal,
            marker_label="101",
            posArc=np.array([0.1, 0.2, 0.3]),
            orArc=np.zeros(3),
        )

        self.assertEqual(tree.pathToGoal.ids[2], "101")
        np.testing.assert_allclose(tree.pathToGoal.tfs[2][:3, 3], [0.1, 0.2, 0.3])

    def test_apply_chain_order_with_rotation(self):
        path = {
            "ids": ["A", "B", "C"],
            "tfs": [
                make_transform(np.array([1.0, 0.0, 0.0]), np.array([np.pi / 2, 0.0, 0.0])),
                make_transform(np.array([2.0, 0.0, 0.0]), np.zeros(3)),
                make_transform(np.array([0.0, 1.0, 0.0]), np.zeros(3)),
            ],
        }
        composed = apply_chain(path)
        expected = path["tfs"][0] @ path["tfs"][1] @ path["tfs"][2]
        np.testing.assert_allclose(composed, expected)
        self.assertFalse(np.allclose(composed[:3, 3], [3.0, 1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
