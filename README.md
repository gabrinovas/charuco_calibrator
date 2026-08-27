# Charuco Calibrator

`charuco_calibrator` is a ROS 2 package designed to facilitate both **intrinsic camera calibration** and **extrinsic hand-eye calibration** using ChArUco boards. It provides a set of nodes, scripts, and launch files to capture images, detect markers, generate calibration pairs, and interface seamlessly with `visp_hand2eye_calibration`.

## Features
- **Multi-Robot Platform Support**: Out-of-the-box parameterization for **Universal Robots UR5e** (`ur5e`, `base` / `tool0`) and **UFactory Lite 6** (`ufactory_lite6`, `link_base` / `link_eef`), as well as customizable robot models.
- **Intrinsic Calibration**: Capture images and calibrate your camera's intrinsics (camera matrix and distortion coefficients) using a ChArUco board.
- **Hand-Eye Calibration**: Synchronize and process robot poses and ChArUco board detections to compute transformations for eye-in-hand and eye-to-hand setups.
- **Data Capture Tools**: Helper scripts to capture images from camera streams and save robot TF poses (`base` / `link_base` to tool frame) via ROS 2 services.
- **VISP Integration**: Automatically publishes corresponding pose pairs (`world_effector` and `camera_object`) required by the `visp_hand2eye_calibration` solver.

---

## Package Structure

### 🚀 Launch Files (`/launch`)
- **`charuco_detector.launch.py`**: Launches the intrinsic camera calibration node.
- **`hand_eye_calibrator.launch.py`**: Launches the offline hand-eye calibration node with parameters for robot model and directory paths.

### ⚙️ Configuration (`/config`)
- **`charuco_params.yaml`**: Main configuration file containing parameters for the ChArUco board (dimensions, square/marker sizes, dictionary), camera resolution, and output directories.

### 🧠 Nodes (`/charuco_calibrator`)
- **`charuco_intrinsic.py`** (`charuco_intrinsic`): Node that processes a folder of captured images to compute the camera matrix and distortion coefficients.
- **`charuco_hand_eye.py`** (`charuco_hand_eye`): Offline detection node. It reads paired images and robot poses, detects the board, and publishes the transformations to `/world_effector_poses` and `/camera_object_poses` for VISP. Supports `-p robot_name:=<ur5e|ufactory_lite6>` and `-p base_frame:=<base|link_base>`.

### 📜 Helper Scripts (`/scripts`)
- **`capture_for_calibration.py`**: A standalone OpenCV script to capture and save images (manually via SPACE or continuously).
- **`save_robot_pose.py`**: A ROS 2 node that listens to TF and provides a `~/save_pose` service (Trigger) to dump the current robot pose (`base` / `link_base` to tool frame) to YAML/TXT files.
- **`generate_calibration_pairs.py`**: A script that iterates over a dataset of images and robot poses, runs the ChArUco pose estimation, and saves perfectly paired YAML datasets.

---

## Supported Robots & Default Frames

| Robot Model | `robot_name` Parameter | Base Frame (`base_frame`) | End-Effector Frame (`tool_frame`) |
| :--- | :--- | :--- | :--- |
| **UR5e** | `ur5e` | `base` | `tool0` |
| **UFactory Lite 6** | `ufactory_lite6` | `link_base` | `link_eef` |

---

## Dependencies
- ROS 2 (tested on Humble / Iron)
- OpenCV (`cv2`) and `cv2.aruco`
- `cv_bridge`
- `tf2_ros` and `tf_transformations`
- `visp_hand2eye_calibration`
- `flexbe_core` & `flexbe_msgs`

---

## Workflow Example: Hand-Eye Calibration

1. **Configure your setup**: Edit `config/charuco_params.yaml` to match your printed ChArUco board specifications and directories.
2. **Capture Intrinsic Data**: Run `capture_for_calibration.py` or the FlexBE camera calibration state machine to capture images of the board from various angles.
3. **Calibrate Intrinsics**: Run `charuco_detector.launch.py` to generate `camera_intrinsics.yaml` (camera matrix and distortion coefficients).
4. **Capture Extrinsic Data**:
   - Move the robot (`ur5e` or `ufactory_lite6`) to various poses.
   - For each pose, capture an image (`capture_for_calibration.py` / `take_pose_and_picture`) and save the robot pose calling the service provided by `save_robot_pose.py`.
5. **Process Pairs & Publish to VISP**: Run `hand_eye_calibrator.launch.py` (or FlexBE behavior) with the selected `robot_name`. It will process detections and publish `/world_effector_poses` and `/camera_object_poses`.
6. **Solve Calibration**: Run the VISP calibrator node (`visp_hand2eye_calibration_calibrator`) to solve $AX=XB$ and export the final transformation matrix (`eye_in_hand_calibration_<robot_name>.yaml`).
