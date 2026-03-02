# Quadcopter localization with ArUco + Kalman filter (OpenCV)

This repository contains a practical starter for your drone setup:
- detect ArUco marker(s) on the drone,
- estimate 3D marker position using OpenCV,
- smooth the position with a Kalman filter.

## Install

```bash
pip install opencv-contrib-python numpy
```

## Which MATLAB files are required?

If your project has many MATLAB files, for this OpenCV pipeline you only need data that defines **camera calibration intrinsics**. You do **not** need flight-control MATLAB code to run this script.

Minimum required MATLAB-origin information:
- `FocalLength = [fx, fy]`
- `PrincipalPoint = [cx, cy]`
- `RadialDistortion = [k1, k2]` (or more radial terms if available)
- `TangentialDistortion = [p1, p2]`
- `ImageSize = [rows, cols]`

Good sources in MATLAB projects are usually one of:
- a calibration `.mat` file containing `cameraParams` / intrinsics,
- a MATLAB script that defines these values,
- generated C++ like your `cameraIntrinsics.cpp/.h` that embeds these constants.

Not required for this script:
- mission planning scripts,
- state-machine logic,
- motor mixing / control allocation,
- telemetry visualization code,
- unrelated Simulink model files.

## Run options

### Option A: use calibration `.npy` files

```bash
python drone_localization.py \
  --camera-id 0 \
  --marker-size 0.12 \
  --dict DICT_4X4_50 \
  --camera-matrix camera_matrix.npy \
  --dist-coeffs dist_coeffs.npy
```

### Option B: use your MATLAB cameraIntrinsics values directly

This repo includes conversion from MATLAB values (`FocalLength`, `PrincipalPoint`, `RadialDistortion`, `TangentialDistortion`) to OpenCV `camera_matrix` and `dist_coeffs`.

```bash
python drone_localization.py \
  --camera-id 0 \
  --marker-size 0.12 \
  --use-matlab-intrinsics
```

## MATLAB → OpenCV mapping used

From your snippet:
- `FocalLength = [fx, fy]`
- `PrincipalPoint = [cx, cy]`
- `RadialDistortion = [k1, k2]`
- `TangentialDistortion = [p1, p2]`

OpenCV uses:
- `camera_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`
- `dist_coeffs = [k1, k2, p1, p2, k3]`

Because your MATLAB code only provides two radial terms, this starter sets `k3 = 0`.

## Important calibration note

Your MATLAB `ImageSize` is `[480, 768]` (`[rows, cols]`).
If your live camera stream is a different resolution, pose results can be biased.
The script prints a warning when resolution mismatch is detected in MATLAB-intrinsics mode.

## Coordinate interpretation

`estimatePoseSingleMarkers` returns marker pose in the camera frame:
- `tvec[0]`: x (right)
- `tvec[1]`: y (down)
- `tvec[2]`: z (forward from camera)

Press `q` to quit.
