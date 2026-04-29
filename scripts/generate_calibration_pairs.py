#!/usr/bin/env python3
"""
Script to generate calibration pairs from images and saved poses.
Run after having all images and robot poses.
"""

import os
import yaml
import cv2
import numpy as np
import cv2.aruco as aruco
import glob
import tf_transformations
from datetime import datetime

def main():
    # Configuración
    PICTURES_FOLDER = "/home/drims/drims_ws/calibrations/extrinsic_calibration/pictures"
    ROBOT_POSES_FOLDER = "/home/drims/drims_ws/calibrations/extrinsic_calibration/robot_poses"
    OUTPUT_FOLDER = "/home/drims/drims_ws/calibrations/extrinsic_calib_charuco_poses"
    CONFIG_FILE = "/home/drims/drims_ws/src/charuco_calibrator/config/charuco_params.yaml"
    
    # Create output folders
    PAIRS_FOLDER = os.path.join(OUTPUT_FOLDER, "pairs")
    DETECTIONS_FOLDER = os.path.join(OUTPUT_FOLDER, "detections")
    os.makedirs(PAIRS_FOLDER, exist_ok=True)
    os.makedirs(DETECTIONS_FOLDER, exist_ok=True)
    
    # Load board configuration
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    calib_params = config.get('charuco_calibrator', {}).get('ros__parameters', {})
    
    rows = calib_params.get('charuco_rows', 14)
    cols = calib_params.get('charuco_cols', 10)
    square_length = calib_params.get('square_length', 0.020)
    marker_length = calib_params.get('marker_length', 0.015)
    dictionary_name = calib_params.get('dictionary', 'DICT_4X4_100')
    
    # Configure detector
    dictionary_map = {
        'DICT_4X4_50': aruco.DICT_4X4_50,
        'DICT_4X4_100': aruco.DICT_4X4_100,
        'DICT_4X4_250': aruco.DICT_4X4_250,
        'DICT_4X4_1000': aruco.DICT_4X4_1000,
        'DICT_5X5_50': aruco.DICT_5X5_50,
        'DICT_5X5_100': aruco.DICT_5X5_100,
        'DICT_5X5_250': aruco.DICT_5X5_250,
        'DICT_5X5_1000': aruco.DICT_5X5_1000,
        'DICT_6X6_50': aruco.DICT_6X6_50,
        'DICT_6X6_100': aruco.DICT_6X6_100,
        'DICT_6X6_250': aruco.DICT_6X6_250,
        'DICT_6X6_1000': aruco.DICT_6X6_1000,
        'DICT_7X7_50': aruco.DICT_7X7_50,
        'DICT_7X7_100': aruco.DICT_7X7_100,
        'DICT_7X7_250': aruco.DICT_7X7_250,
        'DICT_7X7_1000': aruco.DICT_7X7_1000,
        'DICT_ARUCO_ORIGINAL': aruco.DICT_ARUCO_ORIGINAL
    }
    
    dict_id = dictionary_map.get(dictionary_name, aruco.DICT_4X4_100)
    aruco_dict = aruco.getPredefinedDictionary(dict_id)
    
    board = aruco.CharucoBoard(
        (cols, rows),
        square_length,
        marker_length,
        aruco_dict
    )
    
    # Load camera intrinsics (assuming they are in pictures_folder)
    intrinsics_file = os.path.join(PICTURES_FOLDER, 'calibration.yaml')
    if os.path.exists(intrinsics_file):
        with open(intrinsics_file, 'r') as f:
            intrinsics = yaml.safe_load(f)
        camera_matrix = np.array(intrinsics['camera_matrix'])
        dist_coeffs = np.array(intrinsics['distortion_coefficients'])
    else:
        print("⚠️ Intrinsics file not found, using default values")
        camera_matrix = np.eye(3)
        dist_coeffs = np.zeros(5)
    
    # Search images
    image_paths = sorted(glob.glob(os.path.join(PICTURES_FOLDER, '*.jpg')) +
                         glob.glob(os.path.join(PICTURES_FOLDER, '*.png')))
    
    # Search robot poses
    pose_files = sorted(glob.glob(os.path.join(ROBOT_POSES_FOLDER, 'pose_*.yaml')))
    
    print(f"📸 Images found: {len(image_paths)}")
    print(f"🤖 Poses found: {len(pose_files)}")
    
    # Process each pair
    valid_pairs = 0
    
    for i, (image_path, pose_file) in enumerate(zip(image_paths, pose_files)):
        print(f"\n📌 Processing pair {i+1}:")
        print(f"   Image: {os.path.basename(image_path)}")
        print(f"   Pose: {os.path.basename(pose_file)}")
        
        # Read robot pose
        with open(pose_file, 'r') as f:
            pose_data = yaml.safe_load(f)
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"   ❌ Could not read the image")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
        
        if ids is None or len(ids) < 4:
            print(f"   ❌ Not enough markers detected")
            continue
        
        # Interpolate Charuco corners
        ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        
        if charuco_corners is None or len(charuco_corners) < 10:
            print(f"   ❌ Not enough Charuco corners")
            continue
        
        # Estimate pose
        ret, rvec, tvec = aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board,
            camera_matrix, dist_coeffs, None, None
        )
        
        if not ret:
            print(f"   ❌ Could not estimate pose")
            continue
        
        # Convert to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Get quaternion
        T = np.vstack([np.hstack([R, tvec.reshape(3, 1)]), [0, 0, 0, 1]])
        quat = tf_transformations.quaternion_from_matrix(T)
        
        # Save individual detection
        detection_data = {
            'image': os.path.basename(image_path),
            'timestamp': datetime.now().isoformat(),
            'translation': tvec.flatten().tolist(),
            'rotation_matrix': R.tolist(),
            'quaternion': quat.tolist(),
            'num_corners': len(charuco_corners),
            'num_markers': len(ids)
        }
        
        detection_file = os.path.join(DETECTIONS_FOLDER, f"detection_{i+1:03d}.yaml")
        with open(detection_file, 'w') as f:
            yaml.dump(detection_data, f, default_flow_style=False)
        
        # Save complete pair
        pair_data = {
            'index': i+1,
            'image_file': os.path.basename(image_path),
            'pose_file': os.path.basename(pose_file),
            'detection_file': f"detection_{i+1:03d}.yaml",
            'robot_pose': {
                'position': pose_data['position'],
                'orientation': pose_data['orientation']
            },
            'charuco_pose': {
                'translation': tvec.flatten().tolist(),
                'quaternion': quat.tolist()
            },
            'num_corners': len(charuco_corners)
        }
        
        pair_file = os.path.join(PAIRS_FOLDER, f"pair_{i+1:03d}.yaml")
        with open(pair_file, 'w') as f:
            yaml.dump(pair_data, f, default_flow_style=False)
        
        # Visualization
        img_viz = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
        img_viz = cv2.aruco.drawDetectedCornersCharuco(img_viz, charuco_corners, charuco_ids)
        cv2.drawFrameAxes(img_viz, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
        
        viz_path = os.path.join(OUTPUT_FOLDER, f"viz_{i+1:03d}.jpg")
        cv2.imwrite(viz_path, img_viz)
        
        valid_pairs += 1
        print(f"   ✅ Valid pair #{valid_pairs} ({len(charuco_corners)} corners)")
    
    # Save summary
    summary = {
        'total_images': len(image_paths),
        'total_poses': len(pose_files),
        'valid_pairs': valid_pairs,
        'timestamp': datetime.now().isoformat(),
        'charuco_config': {
            'rows': rows,
            'cols': cols,
            'square_length': square_length,
            'marker_length': marker_length,
            'dictionary': dictionary_name
        }
    }
    
    summary_file = os.path.join(OUTPUT_FOLDER, 'calibration_summary.yaml')
    with open(summary_file, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    print(f"\n{'='*50}")
    print(f"✅ Processing completed")
    print(f"📊 Valid pairs: {valid_pairs}/{len(image_paths)}")
    print(f"📁 Detections: {DETECTIONS_FOLDER}")
    print(f"📁 Pairs: {PAIRS_FOLDER}")
    print(f"📄 Summary: {summary_file}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
