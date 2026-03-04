#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import cv2.aruco as aruco
import os
import glob
import yaml
from geometry_msgs.msg import Transform, TransformStamped
from visp_hand2eye_calibration.msg import TransformArray
from std_msgs.msg import Header
from ament_index_python.packages import get_package_share_directory
import tf_transformations
import time

class HandEyeCalibrator(Node):
    """
    Nodo de detección offline de Charuco para calibración ojo-mano.
    Lee imágenes guardadas y publica las poses del tablero para VISP.
    """
    
    def __init__(self):
        super().__init__('hand_eye_calibrator')
        
        # Parámetros
        self.declare_parameter('pictures_folder', '/home/drims/drims_ws/calibrations/extrinsic_calibration/pictures')
        self.declare_parameter('robot_poses_folder', '/home/drims/drims_ws/calibrations/extrinsic_calibration/robot_poses')
        self.declare_parameter('config_file', 'charuco_params.yaml')
        self.declare_parameter('camera_intrinsics_file', 'calibration.yaml')
        self.declare_parameter('eye_in_hand', False)
        self.declare_parameter('publish_rate', 1.0)  # Hz para publicar
        
        # Obtener parámetros
        self.pictures_folder = self.get_parameter('pictures_folder').value
        self.robot_poses_folder = self.get_parameter('robot_poses_folder').value
        self.config_file = self.get_parameter('config_file').value
        self.camera_intrinsics_file = self.get_parameter('camera_intrinsics_file').value
        self.eye_in_hand = self.get_parameter('eye_in_hand').value
        self.publish_rate = self.get_parameter('publish_rate').value
        
        # Publicadores para VISP
        self.world_effector_pub = self.create_publisher(TransformArray, '/world_effector_poses', 10)
        self.camera_object_pub = self.create_publisher(TransformArray, '/camera_object_poses', 10)
        
        # Timer para publicación
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)
        
        # Cargar configuración
        self.load_config()
        
        # Cargar intrínsecos
        self.load_camera_intrinsics()
        
        # Configurar detector
        self.setup_charuco_board()
        
        # Almacenar detecciones
        self.detections = []
        self.robot_poses = []
        self.calibration_pairs = []
        self.processed = False
        
        self.get_logger().info("="*50)
        self.get_logger().info("🔍 DETECTOR OFFLINE CHARUCO")
        self.get_logger().info("="*50)
        self.get_logger().info(f"📁 Imágenes: {self.pictures_folder}")
        self.get_logger().info(f"📁 Poses robot: {self.robot_poses_folder}")
        self.get_logger().info(f"🎯 Modo: {'Eye-in-hand' if self.eye_in_hand else 'Eye-to-hand'}")
        
        # Procesar imágenes al iniciar
        self.process_images()

    def load_config(self):
        """Carga configuración del tablero"""
        # Buscar archivo de configuración
        possible_paths = [
            self.config_file,
            os.path.join(get_package_share_directory('charuco_calibrator'), 'config', self.config_file),
            os.path.join('/home/drims/drims_ws/src/charuco_calibrator/config', self.config_file)
        ]
        
        config_path = None
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path is None:
            raise FileNotFoundError(f"❌ No se encontró archivo de configuración: {self.config_file}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        calib_params = config.get('charuco_calibrator', {}).get('ros__parameters', {})
        
        self.rows = calib_params.get('charuco_rows', 14)
        self.cols = calib_params.get('charuco_cols', 10)
        self.square_length = calib_params.get('square_length', 0.020)
        self.marker_length = calib_params.get('marker_length', 0.015)
        self.dictionary_name = calib_params.get('dictionary', 'DICT_4X4_100')
        
        self.get_logger().info(f"📋 Tablero: {self.cols}x{self.rows}, {self.square_length*1000:.1f}mm")

    def load_camera_intrinsics(self):
        """Carga calibración intrínseca de la cámara"""
        # Buscar en ubicaciones comunes
        search_paths = [
            self.camera_intrinsics_file,
            os.path.join('/home/drims/drims_ws/calibrations/intrinsic_calibration', self.camera_intrinsics_file),
            os.path.join('/home/drims/drims_ws/calibrations', self.camera_intrinsics_file)
        ]
        
        intrinsics_path = None
        for path in search_paths:
            if os.path.exists(path):
                intrinsics_path = path
                break
        
        if intrinsics_path is None:
            raise FileNotFoundError(f"❌ No se encuentra archivo de intrínsecos: {self.camera_intrinsics_file}")
        
        with open(intrinsics_path, 'r') as f:
            data = yaml.safe_load(f)
        
        self.camera_matrix = np.array(data['camera_matrix'])
        self.dist_coeffs = np.array(data['distortion_coefficients'])
        
        self.get_logger().info(f"✅ Intrínsecos cargados: {os.path.basename(intrinsics_path)}")

    def setup_charuco_board(self):
        """Configura el detector Charuco"""
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
        
        dict_id = dictionary_map.get(self.dictionary_name, aruco.DICT_4X4_100)
        self.aruco_dict = aruco.getPredefinedDictionary(dict_id)
        
        self.board = aruco.CharucoBoard(
            (self.cols, self.rows),
            self.square_length,
            self.marker_length,
            self.aruco_dict
        )
        
        self.detector_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.detector_params)

    def load_robot_poses(self):
        """Carga las poses del robot desde archivos"""
        pose_files = sorted(glob.glob(os.path.join(self.robot_poses_folder, 'pose_*.yaml')))
        
        robot_poses = []
        
        for pose_file in pose_files:
            try:
                with open(pose_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                # Extraer índice del nombre
                index = int(os.path.basename(pose_file).split('_')[1].split('.')[0])
                
                robot_poses.append({
                    'file': pose_file,
                    'index': index,
                    'position': np.array(data['position']),
                    'orientation': np.array(data['orientation']),
                    'timestamp': data.get('timestamp', 0)
                })
                
            except Exception as e:
                self.get_logger().warn(f"⚠️ Error cargando {pose_file}: {e}")
        
        self.get_logger().info(f"✅ Cargadas {len(robot_poses)} poses del robot")
        return robot_poses

    def detect_board_in_image(self, image_path):
        """Detecta el tablero Charuco en una imagen"""
        img = cv2.imread(image_path)
        if img is None:
            return None, None, None, None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detectar marcadores
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict)
        
        if ids is None or len(ids) < 4:
            return None, None, None, None
        
        # Interpolar esquinas Charuco
        ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            corners, ids, gray, self.board
        )
        
        if charuco_corners is None or len(charuco_corners) < 4:
            return None, None, None, None
        
        # Estimar pose
        ret, rvec, tvec = aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, self.board,
            self.camera_matrix, self.dist_coeffs, None, None
        )
        
        if not ret:
            return None, None, None, None
        
        # Crear imagen de visualización (opcional)
        img_viz = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
        img_viz = cv2.aruco.drawDetectedCornersCharuco(img_viz, charuco_corners, charuco_ids)
        cv2.drawFrameAxes(img_viz, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)
        
        # Guardar visualización en la misma carpeta
        viz_path = image_path.replace('.jpg', '_detected.jpg').replace('.png', '_detected.png')
        cv2.imwrite(viz_path, img_viz)
        
        return rvec, tvec, len(charuco_corners), viz_path

    def extract_index_from_filename(self, filename):
        """Extrae índice numérico del nombre de archivo"""
        import re
        basename = os.path.basename(filename)
        match = re.search(r'image_(\d+)', basename)
        if match:
            return int(match.group(1))
        return None

    def process_images(self):
        """Procesa todas las imágenes y empareja con poses del robot"""
        
        # 1. Cargar poses del robot
        self.robot_poses = self.load_robot_poses()
        if not self.robot_poses:
            self.get_logger().error("❌ No hay poses del robot")
            return
        
        # 2. Buscar imágenes
        image_paths = sorted(glob.glob(os.path.join(self.pictures_folder, 'image_*.jpg')) +
                            glob.glob(os.path.join(self.pictures_folder, 'image_*.png')))
        
        if not image_paths:
            self.get_logger().error(f"❌ No hay imágenes en {self.pictures_folder}")
            return
        
        self.get_logger().info(f"📸 Procesando {len(image_paths)} imágenes...")
        
        # 3. Crear mapa de poses por índice
        pose_map = {pose['index']: pose for pose in self.robot_poses}
        
        # 4. Procesar cada imagen
        for image_path in image_paths:
            img_index = self.extract_index_from_filename(image_path)
            
            if img_index is None:
                self.get_logger().warn(f"⚠️ No se pudo extraer índice de {os.path.basename(image_path)}")
                continue
            
            # Buscar pose correspondiente
            if img_index not in pose_map:
                self.get_logger().warn(f"⚠️ No hay pose para índice {img_index}")
                continue
            
            pose = pose_map[img_index]
            
            # Detectar charuco
            self.get_logger().info(f"   Procesando {os.path.basename(image_path)} (índice {img_index})...")
            rvec, tvec, num_corners, viz_path = self.detect_board_in_image(image_path)
            
            if rvec is None:
                self.get_logger().warn(f"   ⚠️ No se detectó tablero")
                continue
            
            # Convertir a matrices
            R_board_in_cam, _ = cv2.Rodrigues(rvec)
            t_board_in_cam = tvec.reshape(3, 1)
            
            # Guardar detección
            detection = {
                'index': img_index,
                'image': image_path,
                'visualization': viz_path,
                'num_corners': num_corners,
                'R_board_in_cam': R_board_in_cam,
                't_board_in_cam': t_board_in_cam,
                'rvec': rvec.flatten().tolist(),
                'tvec': tvec.flatten().tolist()
            }
            self.detections.append(detection)
            
            # Crear par de calibración
            pair = {
                'index': img_index,
                'robot_pose': pose,
                'charuco_detection': detection
            }
            self.calibration_pairs.append(pair)
            
            self.get_logger().info(f"   ✅ Detectado con {num_corners} esquinas")
        
        # 5. Guardar detecciones en archivo
        self.save_detections()
        
        self.processed = True
        self.get_logger().info(f"\n✅ Procesamiento completado: {len(self.calibration_pairs)} pares válidos")
        self.get_logger().info(f"📊 Total imágenes: {len(image_paths)}, Detecciones exitosas: {len(self.detections)}")

    def save_detections(self):
        """Guarda las detecciones en archivo YAML"""
        if not self.detections:
            return
        
        # Guardar en la carpeta de calibraciones
        output_file = '/home/drims/drims_ws/calibrations/charuco_detections.yaml'
        
        # Convertir numpy arrays a listas para YAML
        detections_data = []
        for d in self.detections:
            det_data = {
                'index': d['index'],
                'image': os.path.basename(d['image']),
                'visualization': os.path.basename(d['visualization']),
                'num_corners': d['num_corners'],
                'translation': d['tvec'],
                'rvec': d['rvec']
            }
            detections_data.append(det_data)
        
        data = {
            'timestamp': time.time(),
            'eye_in_hand': self.eye_in_hand,
            'num_detections': len(self.detections),
            'detections': detections_data
        }
        
        with open(output_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        self.get_logger().info(f"💾 Detecciones guardadas en: {output_file}")

    def timer_callback(self):
        """Publica los pares de calibración para VISP"""
        if not self.processed or len(self.calibration_pairs) == 0:
            return
        
        # Crear mensajes
        world_effector_msg = TransformArray()
        camera_object_msg = TransformArray()
        
        world_effector_msg.header = Header()
        world_effector_msg.header.stamp = self.get_clock().now().to_msg()
        world_effector_msg.header.frame_id = 'base_link'
        
        camera_object_msg.header = Header()
        camera_object_msg.header.stamp = self.get_clock().now().to_msg()
        camera_object_msg.header.frame_id = 'camera_color_optical_frame'
        
        for pair in self.calibration_pairs:
            # Transform base → tool (pose del robot)
            trans_robot = Transform()
            pos = pair['robot_pose']['position']
            quat = pair['robot_pose']['orientation']
            
            trans_robot.translation.x = float(pos[0])
            trans_robot.translation.y = float(pos[1])
            trans_robot.translation.z = float(pos[2])
            trans_robot.rotation.x = float(quat[0])
            trans_robot.rotation.y = float(quat[1])
            trans_robot.rotation.z = float(quat[2])
            trans_robot.rotation.w = float(quat[3])
            
            # Transform cámara → charuco (detección)
            trans_charuco = Transform()
            t = pair['charuco_detection']['t_board_in_cam'].flatten()
            R = pair['charuco_detection']['R_board_in_cam']
            
            # Convertir matriz de rotación a cuaternión
            T = np.eye(4)
            T[:3, :3] = R
            quat_charuco = tf_transformations.quaternion_from_matrix(T)
            
            trans_charuco.translation.x = float(t[0])
            trans_charuco.translation.y = float(t[1])
            trans_charuco.translation.z = float(t[2])
            trans_charuco.rotation.x = float(quat_charuco[0])
            trans_charuco.rotation.y = float(quat_charuco[1])
            trans_charuco.rotation.z = float(quat_charuco[2])
            trans_charuco.rotation.w = float(quat_charuco[3])
            
            # Añadir a los mensajes
            world_effector_msg.transforms.append(trans_robot)
            camera_object_msg.transforms.append(trans_charuco)
        
        # Publicar
        self.world_effector_pub.publish(world_effector_msg)
        self.camera_object_pub.publish(camera_object_msg)
        
        self.get_logger().info(f"📢 Publicados {len(self.calibration_pairs)} pares para VISP", throttle_duration_sec=5.0)

def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCalibrator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Nodo detenido por usuario")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
