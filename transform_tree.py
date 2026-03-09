"""Transform tree compatibility layer for MATLAB-generated initTree/detectGoal.

This module mirrors the tree update semantics from generated C++ code while
remaining easy to call from Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np


@dataclass
class PathChain:
    """One ordered transform chain.

    ids: node labels (e.g., ["UAV", "CAM", "101"])
    tfs: corresponding 4x4 transforms
    """

    ids: List[str]
    tfs: List[np.ndarray]


@dataclass
class TransformTree:
    """Container with two path branches used by detectGoal."""

    pathToGoal: PathChain
    pathToStart: PathChain


def eul2rotm_zyx(or_zyx: np.ndarray) -> np.ndarray:
    """Euler ZYX (yaw,pitch,roll) to rotation matrix."""
    z, y, x = np.asarray(or_zyx, dtype=np.float64)
    cz, sz = np.cos(z), np.sin(z)
    cy, sy = np.cos(y), np.sin(y)
    cx, sx = np.cos(x), np.sin(x)

    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


def make_transform(position_xyz: np.ndarray, eul_zyx: np.ndarray) -> np.ndarray:
    """Build homogeneous 4x4 transform from position + ZYX Euler."""
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = eul2rotm_zyx(eul_zyx)
    tf[:3, 3] = np.asarray(position_xyz, dtype=np.float64).reshape(3)
    return tf


def _normalize_node_id(node_id: Any) -> str:
    """Normalize incoming IDs to strings (works with char arrays / ints)."""
    return str(node_id).strip()


def _to_matrix(tf_like: Any) -> np.ndarray:
    """Convert generic transform container into 4x4 numpy matrix."""
    return np.asarray(tf_like, dtype=np.float64).reshape(4, 4)


def _get_paths(tree: Any) -> tuple[Any, Any]:
    """Access pathToGoal/pathToStart from object or dict tree."""
    if hasattr(tree, "pathToGoal") and hasattr(tree, "pathToStart"):
        return tree.pathToGoal, tree.pathToStart
    if isinstance(tree, dict) and "pathToGoal" in tree and "pathToStart" in tree:
        return tree["pathToGoal"], tree["pathToStart"]
    raise TypeError("tree must provide pathToGoal and pathToStart")


def _get_ids(path: Any) -> list[str]:
    """Read path IDs independent of object/dict representation."""
    ids = path.ids if hasattr(path, "ids") else path["ids"]
    return [_normalize_node_id(v) for v in ids]


def _get_tfs(path: Any) -> list[np.ndarray]:
    """Read path transforms independent of object/dict representation."""
    tfs = path.tfs if hasattr(path, "tfs") else path["tfs"]
    return [_to_matrix(tf) for tf in tfs]


def _set_tf(path: Any, idx: int, tf: np.ndarray) -> None:
    """Write transform into path at index idx."""
    if hasattr(path, "tfs"):
        path.tfs[idx] = tf
    else:
        path["tfs"][idx] = tf


def _set_id(path: Any, idx: int, node_id: str) -> None:
    """Write node ID into path at index idx."""
    if hasattr(path, "ids"):
        path.ids[idx] = node_id
    else:
        path["ids"][idx] = node_id


def _id3(node_id: str) -> str:
    """Match MATLAB-generated 3-char ID behavior (pad/truncate to 3).

    Note: IDs longer than 3 characters intentionally collide with their first
    3 chars (e.g., "1010" -> "101") to preserve MATLAB-generated semantics.
    """
    return (node_id + "   ")[:3]


def initTree(
    posCam: np.ndarray,
    orCam: np.ndarray,
    posUAV: np.ndarray,
    orUAV: np.ndarray,
) -> TransformTree:
    """MATLAB-compatible initTree replacement.

    Initializes both branches with the same sequence:
      [UAV transform] -> [CAM transform] -> [placeholder marker transform]
    """
    ident = np.eye(4, dtype=np.float64)
    uav_tf = make_transform(posUAV, orUAV)
    cam_tf = make_transform(posCam, orCam)

    # Goal branch starts with UAV->CAM->000.
    goal = PathChain(
        ids=["UAV", "CAM", "000"],
        tfs=[uav_tf, cam_tf, ident.copy()],
    )

    # Start branch is initialized as copy of goal branch.
    start = PathChain(
        ids=["UAV", "CAM", "000"],
        tfs=[uav_tf.copy(), cam_tf.copy(), ident.copy()],
    )

    return TransformTree(pathToGoal=goal, pathToStart=start)


def update_uav_transform(path: Any, posUAV: np.ndarray, orUAV: np.ndarray) -> None:
    """Update the transform for node "UAV" in a path."""
    ids = _get_ids(path)
    for i, node_id in enumerate(ids):
        if _id3(node_id) == "UAV":
            _set_tf(path, i, make_transform(posUAV, orUAV))
            return


def updateTreeAfter(path: Any, marker_label: str, posArc: np.ndarray, orArc: np.ndarray) -> None:
    """MATLAB-compatible updateTreeAfter logic.

    Matches generated C++ algorithm:
      1) Search first 2 entries for "CAM".
      2) Write new marker ID into next entry.
      3) Find that marker ID among first 3 entries and update its transform.
    """
    ids = _get_ids(path)
    marker3 = _id3(_normalize_node_id(marker_label))

    # Step 1: search only first two entries for CAM.
    found_cam_at = None
    for i in range(min(2, len(ids))):
        if _id3(ids[i]) == "CAM":
            found_cam_at = i
            break

    if found_cam_at is None:
        return

    # Step 2: write new ID at next slot.
    target_idx = found_cam_at + 1
    if target_idx >= len(ids):
        return
    _set_id(path, target_idx, marker3)

    # Step 3: update transform for node matching marker id.
    ids = _get_ids(path)
    for i, node_id in enumerate(ids[:3]):
        if _id3(node_id) == marker3:
            _set_tf(path, i, make_transform(posArc, orArc))
            return


def apply_chain(path: Any) -> np.ndarray:
    """Multiply all transforms in path order (left-to-right)."""
    transform = np.eye(4, dtype=np.float64)
    for tf in _get_tfs(path):
        transform = transform @ tf
    return transform


def path_for_marker(tree: Any, marker_id: int) -> Any | None:
    """Route known marker IDs to their branch, matching reference switch-case."""
    path_to_goal, path_to_start = _get_paths(tree)
    if marker_id == 101:
        return path_to_goal
    if marker_id == 102:
        return path_to_start
    return None


def init_default_tree() -> TransformTree:
    """Convenience init with MATLAB camera defaults and zero UAV pose."""
    pos_cam = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    or_cam = np.array([np.pi / 2.0, np.pi, 0.0], dtype=np.float64)
    zeros = np.zeros(3, dtype=np.float64)
    return initTree(posCam=pos_cam, orCam=or_cam, posUAV=zeros, orUAV=zeros)
