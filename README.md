# MATLAB-compatible `detectGoal` replacement (OpenCV 4.11)

This repository now targets your exact integration requirement:
- OpenCV: **4.11**
- ArUco marker size: **15 cm x 15 cm** (`0.15 m`)
- Python side replaces MATLAB-generated `detectGoal` and transform-tree logic
- Kalman filter remains in **C++**

## Compatibility with your MATLAB-generated interface

The replacement keeps MATLAB-like naming so it can be wired with minimal changes:

- `detect_goal.py` exposes:
  - `detectGoal(img, tree, posUAV, orUAV, cameraMatrix, distCoeffs, arucoDict=...)`
  - `detect_goal(...)` wrapper for Python-native style
- `transform_tree.py` exposes:
  - `initTree(posCam, orCam, posUAV, orUAV)`
  - `updateTreeAfter(...)`
  - `apply_chain(...)`
  - UAV path update helper (`update_uav_transform`)

Tree compatibility:
- accepts trees with `pathToGoal` and `pathToStart`
- each path supports `ids` and `tfs`
- works for object attributes **or** dict-based structures

## `initTree` behavior matched from your C++ reference

`initTree(...)` now initializes both paths with:
- `ids = ["UAV", "CAM", "000"]`
- `tfs[0] = transform(posUAV, orUAV)`
- `tfs[1] = transform(posCam, orCam)`
- `tfs[2] = identity`

`updateTreeAfter(...)` now mirrors generated C++ behavior exactly: it looks for `"CAM"` in the first two entries, writes the new marker id into the next node, then updates that node transform.

## MATLAB-shape return compatibility

`detectGoal(...)` returns:
- `ids`: shape `(1, N)` (`[[0.0]]` when no detection)
- `arucoPos`: shape `(3, N)` (zeros `(3,1)` when no detection)

This follows MATLAB coder-like array orientation more closely than row-wise Python conventions.

## Files

- `detect_goal.py`: direct detect-goal replacement
- `transform_tree.py`: transform-tree replacement helpers
- `drone_localization.py`: optional standalone local demo/prototype

## Install

```bash
pip install "opencv-contrib-python==4.11.*" numpy
```

## Minimal usage

```python
import cv2
import numpy as np
from detect_goal import detectGoal
from transform_tree import initTree

cameraMatrix = np.array([
    [603.816409588989, 0.0, 388.676982266589],
    [0.0, 600.17202631038163, 240.555861086624],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

distCoeffs = np.array([[0.043102806701415232, -0.1082637934138599, 0.0, 0.0, 0.0]], dtype=np.float64)

posCam = np.array([0.0, 0.0, 0.0])
orCam = np.array([0.0, 0.0, 0.0])
posUAV = np.array([0.0, 0.0, 0.0])
orUAV = np.array([0.0, 0.0, 0.0])

# Direct replacement for initTree step
# (mirrors ids: UAV -> CAM -> 000)
tree = initTree(posCam, orCam, posUAV, orUAV)

img = cv2.imread("frame.png")
ids, arucoPos = detectGoal(img, tree, posUAV, orUAV, cameraMatrix, distCoeffs)
print(ids)
print(arucoPos)
```

## Transform-tree location requested

In this repo, the transform-tree replacement is located in `transform_tree.py`:
- `initTree(...)`
- `updateTreeAfter(...)`
- `apply_chain(...)`

## C++ Kalman boundary

Per your requirement, Python returns measurement outputs (`arucoPos`) and your C++ side should consume them for Kalman prediction/correction.
