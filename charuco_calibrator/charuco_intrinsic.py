#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import cv2.aruco as aruco
import os
import glob
from cv_bridge import CvBridge
import yaml
from ament_index_python.packages import get_package_share_directory

class CharucoIntrinsicCalibrator(Node):
    def __init__(self):
        super().__init__('charuco_intrinsic_calibrator')
        
        # Parameters that can be passed by command line
        self.declare_parameter('images_folder', '')
        self.declare_parameter('config_file', 'charuco_params.yaml')
        self.declare_parameter('output_file', 'camera_intrinsics.yaml')
        
        # Get parameters
        self.images_folder = self.get_parameter('images_folder').value
        config_file = self.get_parameter('config_file').value
        self.output_file = self.get_parameter('output_file').value
        
        # Load configuration from YAML
        self.load_config(config_file)
        
        # Create Charuco dictionary
        self.setup_charuco_board()
        
        self.bridge = CvBridge()
        self.calibrate_from_folder()

    def load_config(self, config_file):
        """Loads configuration from YAML file"""
        # Search for the configuration file
        possible_paths = [
            config_file,
            os.path.join(get_package_share_directory('charuco_calibrator'), 'config', config_file),
            os.path.join(os.path.expanduser('~'), 'ros2_ws', 'src', 'charuco_calibrator', 'config', config_file),
            os.path.join(os.path.dirname(__file__), '..', 'config', config_file)
        ]
        
        config_path = None
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path is None:
            self.get_logger().error(f"❌ Configuration file not found: {config_file}")
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract parameters from charuco_calibrator
        calib_params = config.get('charuco_calibrator', {}).get('ros__parameters', {})
        
        self.rows = calib_params.get('charuco_rows', 14)
        self.cols = calib_params.get('charuco_cols', 10)
        self.square_length = calib_params.get('square_length', 0.020)
        self.marker_length = calib_params.get('marker_length', 0.015)
        self.dictionary_name = calib_params.get('dictionary', 'DICT_4X4_100')
        # Do not use the output_file from config, use the one from parameter
        
        self.get_logger().info(f"📋 Configuration loaded from: {config_path}")
        self.get_logger().info(f"   Board: {self.cols}x{self.rows}")
        self.get_logger().info(f"   Square: {self.square_length}m, Marker: {self.marker_length}m")
        self.get_logger().info(f"   Dictionary: {self.dictionary_name}")

    def setup_charuco_board(self):
        """Configures the Charuco board according to parameters"""
        # Mapping dictionary names to OpenCV objects
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
        
        try:
            dict_id = dictionary_map.get(self.dictionary_name, aruco.DICT_4X4_100)
            self.aruco_dict = aruco.getPredefinedDictionary(dict_id)
        except:
            self.get_logger().warn(f"⚠️ Could not load {self.dictionary_name}, using DICT_4X4_100")
            self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
        
        # Create the board
        self.board = aruco.CharucoBoard(
            (self.cols, self.rows),
            self.square_length,
            self.marker_length,
            self.aruco_dict
        )

    def calibrate_from_folder(self):
        """Calibrates the camera using images from a folder"""
        
        if not self.images_folder:
            self.get_logger().error("❌ images_folder not specified")
            return
        
        if not os.path.exists(self.images_folder):
            self.get_logger().error(f"❌ Folder does not exist: {self.images_folder}")
            return
        
        # Search images
        image_paths = glob.glob(os.path.join(self.images_folder, '*.jpg')) + \
                      glob.glob(os.path.join(self.images_folder, '*.png'))
        
        if len(image_paths) == 0:
            self.get_logger().error(f"No images found in {self.images_folder}")
            return
        
        self.get_logger().info(f"Found {len(image_paths)} images")
        
        all_corners = []
        all_ids = []
        image_size = None
        valid_images = 0
        
        for image_path in image_paths:
            self.get_logger().info(f"Processing: {os.path.basename(image_path)}")
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if image_size is None:
                image_size = gray.shape[::-1]
            
            # Detect ArUco markers
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict)
            
            if ids is not None and len(ids) > 3:
                ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                    corners, ids, gray, self.board
                )
                
                if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 10:
                    all_corners.append(charuco_corners)
                    all_ids.append(charuco_ids)
                    valid_images += 1
                    self.get_logger().info(f"   ✅ Detected {len(charuco_corners)} corners")
                else:
                    self.get_logger().warn(f"   ⚠️ Few corners detected")
            else:
                self.get_logger().warn(f"   ⚠️ Few markers detected")
        
        if len(all_corners) >= 5:
            self.get_logger().info(f"📊 Calibrating with {len(all_corners)} valid images...")
            
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
                charucoCorners=all_corners,
                charucoIds=all_ids,
                board=self.board,
                imageSize=image_size,
                cameraMatrix=None,
                distCoeffs=None
            )
            
            self.get_logger().info(f"\n{'='*50}")
            self.get_logger().info("✅ CALIBRATION SUCCESSFUL")
            self.get_logger().info(f"{'='*50}")
            self.get_logger().info(f"📏 Reprojection error: {ret:.6f}")
            self.get_logger().info(f"\n📷 Camera matrix:\n{camera_matrix}")
            self.get_logger().info(f"\n📐 Distortion coefficients:\n{dist_coeffs.reshape(-1)}")
            
            # Save calibration
            self.save_calibration(camera_matrix, dist_coeffs, image_size, ret, valid_images)
        else:
            self.get_logger().error(f"❌ Not enough patterns detected. Valid: {len(all_corners)}/5")

    def save_calibration(self, camera_matrix, dist_coeffs, image_size, reprojection_error, valid_images):
        """Saves the calibration parameters in YAML file in ~/calibrations/"""
        
        calibration_folder = os.path.expanduser('~/calibrations')
        os.makedirs(calibration_folder, exist_ok=True)
        
        # The filename will be camera_intrinsics.yaml
        output_path = os.path.join(calibration_folder, 'camera_intrinsics.yaml')
        
        calibration_data = {
            'camera_matrix': camera_matrix.tolist(),
            'distortion_coefficients': dist_coeffs.reshape(-1).tolist(),
            'image_width': image_size[0],
            'image_height': image_size[1],
            'reprojection_error': float(reprojection_error),
            'calibration_date': self.get_clock().now().to_msg().sec,
            'images_processed': valid_images,  # Valid images used
            'images_total': len(glob.glob(os.path.join(self.images_folder, '*.jpg')) + 
                            glob.glob(os.path.join(self.images_folder, '*.png'))),  # Total images
            'charuco_config': {
                'rows': self.rows,
                'cols': self.cols,
                'square_length': self.square_length,
                'marker_length': self.marker_length,
                'dictionary': self.dictionary_name
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(calibration_data, f, default_flow_style=False)
        
        self.get_logger().info(f"💾 Calibration saved in: {output_path}")
        self.get_logger().info(f"📁 Image folder used: {self.images_folder}")

def main(args=None):
    rclpy.init(args=args)
    node = CharucoIntrinsicCalibrator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
