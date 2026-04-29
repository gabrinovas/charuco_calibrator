# Charuco Calibrator

`charuco_calibrator` is a ROS 2 package designed to facilitate both **intrinsic camera calibration** and **extrinsic hand-eye calibration** using ChArUco boards. It provides a set of nodes, scripts, and launch files to capture images, detect markers, generate calibration pairs, and interface seamlessly with `visp_hand2eye_calibration`.

## Features
- **Intrinsic Calibration**: Capture images and calibrate your camera's intrinsics (camera matrix and distortion coefficients) using a ChArUco board.
- **Hand-Eye Calibration**: Synchronize and process robot poses and ChArUco board detections to compute transformations for eye-in-hand and eye-to-hand setups.
- **Data Capture Tools**: Scripts to capture images from camera streams and save robot TF poses via ROS 2 services.
- **VISP Integration**: Automatically publishes corresponding pose pairs (`world_effector` and `camera_object`) required by the `visp_hand2eye_calibration` solver.

---

## Package Structure

### 🚀 Launch Files (`/launch`)
- **`charuco_detector.launch.py`**: Launches the intrinsic camera calibration node.
- **`hand_eye_calibrator.launch.py`**: Launches the offline hand-eye calibration node.

### ⚙️ Configuration (`/config`)
- **`charuco_params.yaml`**: Main configuration file containing parameters for the ChArUco board (dimensions, square/marker sizes, dictionary), camera resolution, and output directories.

### 🧠 Nodes (`/charuco_calibrator`)
- **`charuco_intrinsic.py`** (`charuco_intrinsic`): Node that processes a folder of captured images to compute the camera matrix and distortion coefficients.
- **`charuco_hand_eye.py`** (`charuco_hand_eye_offline`): Offline detection node. It reads paired images and robot poses, detects the board, and publishes the transformations to `/world_effector_poses` and `/camera_object_poses` for VISP.

### 📜 Helper Scripts (`/scripts`)
- **`capture_for_calibration.py`**: A standalone OpenCV script to capture and save images (manually via SPACE or continuously).
- **`save_robot_pose.py`**: A ROS 2 node that listens to TF and provides a `~/save_pose` service (Trigger) to dump the current robot pose (base to tool0) to YAML/TXT files.
- **`generate_calibration_pairs.py`**: A script that iterates over a dataset of images and robot poses, runs the ChArUco pose estimation, and saves perfectly paired YAML datasets.

---

## Dependencies
- ROS 2 (tested on Humble/Iron)
- OpenCV (`cv2`) and `cv2.aruco`
- `cv_bridge`
- `tf2_ros` and `tf_transformations`
- `visp_hand2eye_calibration`
- `flexbe_core` & `flexbe_msgs`

---

## Workflow Example: Hand-Eye Calibration

1. **Configure your setup**: Edit `config/charuco_params.yaml` to match your printed ChArUco board and directories.
2. **Capture Intrinsic Data**: Run `capture_for_calibration.py` to capture images of the board from various angles.
3. **Calibrate Intrinsics**: Run `charuco_detector.launch.py` to generate the `calibration.yaml` (camera matrix).
4. **Capture Extrinsic Data**:
   - Move the robot to various poses.
   - For each pose, capture an image (`capture_for_calibration.py`) and save the robot pose calling the service provided by `save_robot_pose.py`.
5. **Process Pairs**: Run `generate_calibration_pairs.py` to pre-calculate the poses of the board in each image.
6. **Publish to VISP**: Run `hand_eye_calibrator.launch.py` to publish the aligned datasets. Run the VISP calibrator node to solve $AX=XB$ and retrieve the final hand-eye transform.
