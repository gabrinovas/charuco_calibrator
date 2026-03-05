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
        self.declare_parameter('output_folder', '/home/drims/drims_ws/calibrations/extrinsic_calibration/charuco_table_poses')
        self.declare_parameter('config_file', 'charuco_params.yaml')
        self.declare_parameter('camera_intrinsics_file', '/home/drims/drims_ws/calibrations/camera_intrinsics.yaml')
        self.declare_parameter('eye_in_hand', False)
        self.declare_parameter('publish_rate', 1.0)  # Hz para publicar
        self.declare_parameter('save_results', True)  # Guardar resultados en archivo
        
        # Obtener parámetros
        self.pictures_folder = self.get_parameter('pictures_folder').value
        self.robot_poses_folder = self.get_parameter('robot_poses_folder').value
        self.output_folder = self.get_parameter('output_folder').value
        self.config_file = self.get_parameter('config_file').value
        self.camera_intrinsics_file = self.get_parameter('camera_intrinsics_file').value
        self.eye_in_hand = self.get_parameter('eye_in_hand').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.save_results = self.get_parameter('save_results').value
        
        # Crear carpeta de salida si no existe
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Publicadores para VISP
        self.world_effector_pub = self.create_publisher(TransformArray, '/world_effector_poses', 10)
        self.camera_object_pub = self.create_publisher(TransformArray, '/camera_object_poses', 10)
        
        # Timer para publicación
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)
        
        # Cargar configuración
        self.load_config()
        
        # Cargar intrínsecos
        if not self.load_camera_intrinsics():
            self.get_logger().error("❌ No se pudo cargar la calibración intrínseca")
            return
        
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
        self.get_logger().info(f"📁 Salida: {self.output_folder}")
        self.get_logger().info(f"📁 Intrínsecos: {self.camera_intrinsics_file}")
        self.get_logger().info(f"🎯 Modo: {'Eye-in-hand' if self.eye_in_hand else 'Eye-to-hand'}")
        
        # Procesar imágenes al iniciar
        self.process_images()

    def load_config(self):
        """Carga configuración del tablero"""
        # Buscar archivo de configuración
        possible_paths = [
            self.config_file,
            os.path.join(get_package_share_directory('charuco_calibrator'), 'config', self.config_file),
            os.path.join('/home/drims/drims_ws/src/charuco_calibrator/config', self.config_file),
            os.path.join('/home/drims/drims_ws/calibrations', self.config_file)
        ]
        
        config_path = None
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path is None:
            # Usar valores por defecto si no encuentra archivo
            self.get_logger().warn(f"⚠️ No se encontró archivo de configuración, usando valores por defecto")
            self.rows = 14
            self.cols = 10
            self.square_length = 0.020
            self.marker_length = 0.015
            self.dictionary_name = 'DICT_4X4_100'
            return
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Intentar diferentes estructuras posibles
        if 'charuco_calibrator' in config:
            calib_params = config.get('charuco_calibrator', {}).get('ros__parameters', {})
        else:
            calib_params = config
        
        self.rows = calib_params.get('charuco_rows', 14)
        self.cols = calib_params.get('charuco_cols', 10)
        self.square_length = calib_params.get('square_length', 0.020)
        self.marker_length = calib_params.get('marker_length', 0.015)
        self.dictionary_name = calib_params.get('dictionary', 'DICT_4X4_100')
        
        self.get_logger().info(f"📋 Tablero: {self.cols}x{self.rows}, {self.square_length*1000:.1f}mm, diccionario: {self.dictionary_name}")

    def load_camera_intrinsics(self):
        """Carga calibración intrínseca de la cámara"""
        if not os.path.exists(self.camera_intrinsics_file):
            self.get_logger().error(f"❌ No se encuentra archivo de intrínsecos: {self.camera_intrinsics_file}")
            return False
        
        try:
            with open(self.camera_intrinsics_file, 'r') as f:
                data = yaml.safe_load(f)
            
            # Manejar diferentes formatos posibles
            if 'camera_matrix' in data:
                self.camera_matrix = np.array(data['camera_matrix'])
            elif 'camera_matrix' in data.get('camera_matrix', {}):
                self.camera_matrix = np.array(data['camera_matrix']['data']).reshape(3, 3)
            else:
                self.get_logger().error("❌ Formato de archivo de intrínsecos no reconocido - no se encuentra 'camera_matrix'")
                return False
            
            if 'distortion_coefficients' in data:
                self.dist_coeffs = np.array(data['distortion_coefficients'])
            elif 'distortion_coefficients' in data.get('distortion_coefficients', {}):
                self.dist_coeffs = np.array(data['distortion_coefficients']['data'])
            else:
                self.get_logger().warn("⚠️ No se encontraron coeficientes de distorsión, usando ceros")
                self.dist_coeffs = np.zeros((5, 1))
            
            self.get_logger().info(f"✅ Intrínsecos cargados: {os.path.basename(self.camera_intrinsics_file)}")
            self.get_logger().info(f"   Matriz cámara:\n{self.camera_matrix}")
            return True
            
        except Exception as e:
            self.get_logger().error(f"❌ Error cargando intrínsecos: {e}")
            return False

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
        
        self.aruco_dict = aruco.Dictionary_get(dict_id)
        
        # Crear CharucoBoard
        self.board = aruco.CharucoBoard_create(
            self.cols, self.rows,
            self.square_length,
            self.marker_length,
            self.aruco_dict
        )
        
        # Configurar detector
        self.detector_params = aruco.DetectorParameters_create()
        
        self.get_logger().info(f"✅ Detector Charuco configurado para OpenCV 4.5.4")

    def load_robot_poses(self):
        """Carga las poses del robot desde archivos YAML con el formato proporcionado"""
        pose_files = sorted(glob.glob(os.path.join(self.robot_poses_folder, 'pose_*.yaml')))
        
        robot_poses = []
        
        for pose_file in pose_files:
            try:
                with open(pose_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                # Formato según tu ejemplo
                robot_poses.append({
                    'file': pose_file,
                    'index': data['index'],
                    'timestamp': data.get('timestamp', 0),
                    'frame_id': data.get('frame_id', 'base_link_to_tool0'),
                    'position': np.array(data['position']),
                    'orientation': np.array(data['orientation'])
                })
                
            except Exception as e:
                self.get_logger().warn(f"⚠️ Error cargando {pose_file}: {e}")
        
        self.get_logger().info(f"✅ Cargadas {len(robot_poses)} poses del robot")
        return robot_poses

    def detect_board_in_image(self, image_path):
        """Detecta el tablero Charuco en una imagen"""
        img = cv2.imread(image_path)
        if img is None:
            return None, None, None, None, None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detectar marcadores
        corners, ids, _ = aruco.detectMarkers(
            gray, 
            self.aruco_dict, 
            parameters=self.detector_params
        )
        
        if ids is None or len(ids) < 4:
            return None, None, None, None, None
        
        # Interpolar esquinas Charuco
        ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            corners, ids, gray, self.board
        )
        
        if charuco_corners is None or len(charuco_corners) < 4:
            return None, None, None, None, None
        
        # Estimar pose
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.zeros((3, 1), dtype=np.float32)
        
        ret = aruco.estimatePoseCharucoBoard(
            charuco_corners, 
            charuco_ids, 
            self.board, 
            self.camera_matrix, 
            self.dist_coeffs,
            rvec, 
            tvec,
            False
        )
        
        if not ret:
            return None, None, None, None, None
        
        # Crear imagen de visualización
        img_viz = img.copy()
        aruco.drawDetectedMarkers(img_viz, corners, ids)
        aruco.drawDetectedCornersCharuco(img_viz, charuco_corners, charuco_ids)
        cv2.drawFrameAxes(img_viz, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)
        
        # Crear nombre para visualización
        basename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(basename)[0]
        viz_path = os.path.join(self.output_folder, f"{name_without_ext}_detected.jpg")
        cv2.imwrite(viz_path, img_viz)
        
        # Obtener matriz de rotación
        R_board_in_cam, _ = cv2.Rodrigues(rvec)
        
        return rvec, tvec, R_board_in_cam, len(charuco_corners), viz_path

    def extract_index_from_filename(self, filename):
        """Extrae índice numérico del nombre de archivo"""
        import re
        basename = os.path.basename(filename)
        # Buscar patrón image_XX o image_XXX
        match = re.search(r'image_(\d+)', basename)
        if match:
            return int(match.group(1))
        # También buscar números al final
        match = re.search(r'(\d+)', basename)
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
        successful_detections = 0
        for image_path in image_paths:
            img_index = self.extract_index_from_filename(image_path)
            
            if img_index is None:
                self.get_logger().warn(f"⚠️ No se pudo extraer índice de {os.path.basename(image_path)}")
                continue
            
            # Buscar pose correspondiente
            if img_index not in pose_map:
                self.get_logger().warn(f"⚠️ No hay pose para índice {img_index} en {os.path.basename(image_path)}")
                continue
            
            pose = pose_map[img_index]
            
            # Detectar charuco
            self.get_logger().info(f"   Procesando {os.path.basename(image_path)} (índice {img_index})...")
            rvec, tvec, R_board_in_cam, num_corners, viz_path = self.detect_board_in_image(image_path)
            
            if rvec is None:
                self.get_logger().warn(f"   ⚠️ No se detectó tablero en {os.path.basename(image_path)}")
                continue
            
            # Guardar detección
            detection = {
                'index': img_index,
                'image': image_path,
                'visualization': viz_path,
                'num_corners': num_corners,
                'R_board_in_cam': R_board_in_cam.tolist(),
                't_board_in_cam': tvec.flatten().tolist(),
                'rvec': rvec.flatten().tolist(),
                'tvec': tvec.flatten().tolist(),
                'timestamp': time.time()
            }
            self.detections.append(detection)
            
            # Crear par de calibración
            pair = {
                'index': img_index,
                'robot_pose': {
                    'position': pose['position'].tolist(),
                    'orientation': pose['orientation'].tolist(),
                    'timestamp': pose.get('timestamp', 0)
                },
                'charuco_detection': {
                    'translation': tvec.flatten().tolist(),
                    'rotation_matrix': R_board_in_cam.tolist(),
                    'rvec': rvec.flatten().tolist(),
                    'num_corners': num_corners
                }
            }
            self.calibration_pairs.append(pair)
            successful_detections += 1
            
            self.get_logger().info(f"   ✅ Detectado con {num_corners} esquinas")
        
        # 5. Guardar detecciones
        if self.save_results and self.detections:
            self.save_detections()
            self.save_calibration_pairs()
        
        self.processed = True
        self.get_logger().info(f"\n✅ Procesamiento completado: {successful_detections} detecciones exitosas de {len(image_paths)} imágenes")
        self.get_logger().info(f"📊 Total pares de calibración: {len(self.calibration_pairs)}")

    def save_detections(self):
        """Guarda las detecciones en archivo YAML"""
        if not self.detections:
            return
        
        output_file = os.path.join(self.output_folder, 'charuco_detections.yaml')
        
        # Convertir a formato serializable
        detections_data = []
        for d in self.detections:
            det_data = {
                'index': d['index'],
                'image': os.path.basename(d['image']),
                'visualization': os.path.basename(d['visualization']) if d['visualization'] else None,
                'num_corners': d['num_corners'],
                'translation': d['tvec'],
                'rvec': d['rvec'],
                'rotation_matrix': d['R_board_in_cam'],
                'timestamp': d['timestamp']
            }
            detections_data.append(det_data)
        
        data = {
            'timestamp': time.time(),
            'eye_in_hand': self.eye_in_hand,
            'num_detections': len(self.detections),
            'detections': detections_data
        }
        
        with open(output_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)
        
        self.get_logger().info(f"💾 Detecciones guardadas en: {output_file}")

    def save_calibration_pairs(self):
        """Guarda los pares de calibración en formato compatible con VISP"""
        if not self.calibration_pairs:
            return
        
        # Guardar en formato para VISP
        output_file = os.path.join(self.output_folder, 'calibration_pairs.yaml')
        
        data = {
            'timestamp': time.time(),
            'eye_in_hand': self.eye_in_hand,
            'num_pairs': len(self.calibration_pairs),
            'pairs': self.calibration_pairs
        }
        
        with open(output_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)
        
        self.get_logger().info(f"💾 Pares de calibración guardados en: {output_file}")
        
        # También guardar una copia en la ubicación esperada por compute_calib.py
        global_pairs_file = '/home/drims/drims_ws/calibrations/charuco_detections.yaml'
        
        # Convertir a formato esperado por offline_find_charuco.py
        simplified_data = {
            'timestamp': time.time(),
            'eye_in_hand': self.eye_in_hand,
            'num_pairs': len(self.calibration_pairs),
            'pairs': []
        }
        
        for pair in self.calibration_pairs:
            simplified_pair = {
                'index': pair['index'],
                'robot_position': pair['robot_pose']['position'],
                'robot_orientation': pair['robot_pose']['orientation'],
                'charuco_translation': pair['charuco_detection']['translation'],
                'charuco_rotation_matrix': pair['charuco_detection']['rotation_matrix']
            }
            simplified_data['pairs'].append(simplified_pair)
        
        with open(global_pairs_file, 'w') as f:
            yaml.dump(simplified_data, f, default_flow_style=False, sort_keys=False, indent=2)
        
        self.get_logger().info(f"💾 Datos simplificados guardados en: {global_pairs_file}")

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
            t = pair['charuco_detection']['translation']
            R = np.array(pair['charuco_detection']['rotation_matrix'])
            
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
