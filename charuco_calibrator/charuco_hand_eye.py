#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import cv2.aruco as aruco
import os
import glob
import yaml
import json
from datetime import datetime
import tf_transformations
from geometry_msgs.msg import Transform, TransformStamped
from visp_hand2eye_calibration.msg import TransformArray
from std_msgs.msg import Header
from ament_index_python.packages import get_package_share_directory

class HandEyeCalibrator(Node):
    """
    Calibración ojo-mano usando imágenes guardadas y poses del robot
    """
    
    def __init__(self):
        super().__init__('hand_eye_calibrator')
        
        # Parámetros
        self.declare_parameter('pictures_folder', '/home/drims/drims_ws/calibrations/extrinsic_calibration/pictures')
        self.declare_parameter('robot_poses_folder', '/home/drims/drims_ws/calibrations/extrinsic_calibration/robot_poses')
        self.declare_parameter('output_folder', '/home/drims/drims_ws/calibrations/extrinsic_calib_charuco_poses')
        self.declare_parameter('config_file', 'charuco_params.yaml')
        self.declare_parameter('camera_intrinsics_file', 'calibration.yaml')
        self.declare_parameter('publish_for_visp', False)  # Opcional: publicar para VISP
        self.declare_parameter('eye_in_hand', False)
        
        # Obtener parámetros
        self.pictures_folder = self.get_parameter('pictures_folder').value
        self.robot_poses_folder = self.get_parameter('robot_poses_folder').value
        self.output_folder = self.get_parameter('output_folder').value
        self.config_file = self.get_parameter('config_file').value
        self.camera_intrinsics_file = self.get_parameter('camera_intrinsics_file').value
        self.publish_for_visp = self.get_parameter('publish_for_visp').value
        self.eye_in_hand = self.get_parameter('eye_in_hand').value
        
        # Crear carpeta de salida
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Publicadores para VISP (opcional)
        if self.publish_for_visp:
            self.world_effector_pub = self.create_publisher(TransformArray, '/world_effector_poses', 10)
            self.camera_object_pub = self.create_publisher(TransformArray, '/camera_object_poses', 10)
        
        # Cargar configuración
        self.load_config()
        
        # Cargar intrínsecos
        self.load_camera_intrinsics()
        
        # Configurar detector
        self.setup_charuco_board()
        
        self.get_logger().info("="*50)
        self.get_logger().info("🔧 CALIBRACIÓN OJO-MANO")
        self.get_logger().info("="*50)
        self.get_logger().info(f"📁 Imágenes: {self.pictures_folder}")
        self.get_logger().info(f"📁 Poses robot: {self.robot_poses_folder}")
        self.get_logger().info(f"📁 Salida: {self.output_folder}")
        self.get_logger().info(f"🎯 Modo: {'Eye-in-hand' if self.eye_in_hand else 'Eye-to-hand'}")
        
        # Ejecutar calibración
        self.calibrate()

    def load_config(self):
        """Carga configuración del tablero"""
        # Buscar archivo de configuración
        possible_paths = [
            self.config_file,
            os.path.join(get_package_share_directory('charuco_calibrator'), 'config', self.config_file),
            os.path.join(os.path.expanduser('~'), 'drims_ws', 'src', 'charuco_calibrator', 'config', self.config_file)
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
            os.path.join(self.pictures_folder, self.camera_intrinsics_file),
            os.path.join('/home/drims/drims_ws/calibrations/intrinsic_calibration', self.camera_intrinsics_file),
            os.path.join(self.output_folder, self.camera_intrinsics_file)
        ]
        
        intrinsics_path = None
        for path in search_paths:
            if os.path.exists(path):
                intrinsics_path = path
                break
        
        if intrinsics_path is None:
            raise FileNotFoundError(f"❌ No se encuentra archivo de intrínsecos")
        
        with open(intrinsics_path, 'r') as f:
            data = yaml.safe_load(f)
        
        self.camera_matrix = np.array(data['camera_matrix'])
        self.dist_coeffs = np.array(data['distortion_coefficients'])
        
        self.get_logger().info(f"✅ Intrínsecos cargados")

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
        """
        Carga las poses del robot desde archivos en la carpeta robot_poses
        Espera archivos: pose_001.yaml, pose_002.yaml, etc.
        """
        pose_files = sorted(glob.glob(os.path.join(self.robot_poses_folder, 'pose_*.yaml')) +
                           glob.glob(os.path.join(self.robot_poses_folder, 'pose_*.txt')))
        
        robot_poses = []
        
        for pose_file in pose_files:
            try:
                if pose_file.endswith('.yaml'):
                    with open(pose_file, 'r') as f:
                        data = yaml.safe_load(f)
                        if 'position' in data and 'orientation' in data:
                            pos = np.array(data['position'])
                            quat = np.array(data['orientation'])
                        elif 'translation' in data and 'rotation' in data:
                            pos = np.array(data['translation'])
                            quat = np.array(data['rotation'])
                        else:
                            continue
                else:  # .txt
                    with open(pose_file, 'r') as f:
                        values = list(map(float, f.readline().strip().split()))
                        if len(values) >= 7:
                            pos = np.array(values[0:3])
                            quat = np.array(values[3:7])
                        elif len(values) == 6:  # x,y,z,rx,ry,rz
                            pos = np.array(values[0:3])
                            rpy = np.array(values[3:6])
                            quat = tf_transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
                        else:
                            continue
                
                robot_poses.append({
                    'file': pose_file,
                    'position': pos,
                    'orientation': quat,
                    'index': int(os.path.basename(pose_file).split('_')[1].split('.')[0])
                })
                
            except Exception as e:
                self.get_logger().warn(f"⚠️ Error cargando {pose_file}: {e}")
        
        self.get_logger().info(f"✅ Cargadas {len(robot_poses)} poses del robot")
        return robot_poses

    def detect_board_in_image(self, image_path):
        """
        Detecta el tablero Charuco en una imagen y devuelve su pose
        """
        img = cv2.imread(image_path)
        if img is None:
            self.get_logger().error(f"❌ No se pudo leer: {image_path}")
            return None, None, None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detectar marcadores
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict)
        
        if ids is None or len(ids) < 4:
            return None, None, None
        
        # Interpolar esquinas Charuco
        ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            corners, ids, gray, self.board
        )
        
        if charuco_corners is None or len(charuco_corners) < 10:
            return None, None, None
        
        # Estimar pose
        ret, rvec, tvec = aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, self.board,
            self.camera_matrix, self.dist_coeffs, None, None
        )
        
        if not ret:
            return None, None, None
        
        # Crear imagen de visualización
        img_viz = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
        img_viz = cv2.aruco.drawDetectedCornersCharuco(img_viz, charuco_corners, charuco_ids)
        cv2.drawFrameAxes(img_viz, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)
        
        # Guardar visualización
        viz_path = image_path.replace('.jpg', '_detected.jpg').replace('.png', '_detected.png')
        cv2.imwrite(viz_path, img_viz)
        
        return rvec, tvec, len(charuco_corners)

    def calibrate(self):
        """
        Proceso principal de calibración
        """
        # 1. Cargar poses del robot
        robot_poses = self.load_robot_poses()
        if not robot_poses:
            self.get_logger().error("❌ No hay poses del robot")
            return
        
        # 2. Buscar imágenes
        image_paths = sorted(glob.glob(os.path.join(self.pictures_folder, '*.jpg')) +
                            glob.glob(os.path.join(self.pictures_folder, '*.png')))
        
        if not image_paths:
            self.get_logger().error(f"❌ No hay imágenes en {self.pictures_folder}")
            return
        
        self.get_logger().info(f"📸 Encontradas {len(image_paths)} imágenes")
        
        # 3. Emparejar imágenes con poses del robot
        # Asumimos que están en el mismo orden o que podemos emparejar por índice
        calibration_pairs = []
        
        for i, image_path in enumerate(image_paths):
            self.get_logger().info(f"📸 Procesando imagen {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            # Detectar pose del tablero
            rvec, tvec, num_corners = self.detect_board_in_image(image_path)
            
            if rvec is None:
                self.get_logger().warn(f"⚠️ No se detectó tablero en {os.path.basename(image_path)}")
                continue
            
            # Buscar pose del robot correspondiente
            # Estrategia 1: por índice en nombre de archivo
            img_index = self.extract_index_from_filename(image_path)
            matching_pose = None
            
            if img_index is not None:
                # Buscar por índice
                for pose in robot_poses:
                    if pose['index'] == img_index:
                        matching_pose = pose
                        break
            
            if matching_pose is None and i < len(robot_poses):
                # Estrategia 2: por orden
                matching_pose = robot_poses[i]
            
            if matching_pose is None:
                self.get_logger().warn(f"⚠️ No se encontró pose del robot para {os.path.basename(image_path)}")
                continue
            
            # Convertir a matrices
            R_board_in_cam, _ = cv2.Rodrigues(rvec)
            t_board_in_cam = tvec.reshape(3, 1)
            
            # Matriz de rotación del robot (base → tool)
            R_base_to_tool = tf_transformations.quaternion_matrix(matching_pose['orientation'])[:3, :3]
            t_base_to_tool = matching_pose['position'].reshape(3, 1)
            
            calibration_pairs.append({
                'image': image_path,
                'robot_pose_file': matching_pose['file'],
                'R_base_to_tool': R_base_to_tool,
                't_base_to_tool': t_base_to_tool,
                'R_board_in_cam': R_board_in_cam,
                't_board_in_cam': t_board_in_cam,
                'num_corners': num_corners
            })
            
            self.get_logger().info(f"   ✅ Par válido #{len(calibration_pairs)} ({num_corners} esquinas)")
        
        if len(calibration_pairs) < 5:
            self.get_logger().error(f"❌ No hay suficientes pares válidos: {len(calibration_pairs)}/5")
            return
        
        self.get_logger().info(f"\n📊 Calibrando con {len(calibration_pairs)} pares...")
        
        # 4. Preparar datos para calibración
        R_base_to_tool_list = [p['R_base_to_tool'] for p in calibration_pairs]
        t_base_to_tool_list = [p['t_base_to_tool'] for p in calibration_pairs]
        R_board_in_cam_list = [p['R_board_in_cam'] for p in calibration_pairs]
        t_board_in_cam_list = [p['t_board_in_cam'] for p in calibration_pairs]
        
        # 5. Ejecutar calibración con diferentes métodos
        results = self.run_calibration_methods(
            R_base_to_tool_list, t_base_to_tool_list,
            R_board_in_cam_list, t_board_in_cam_list
        )
        
        # 6. Guardar resultados
        self.save_calibration_results(results, calibration_pairs)
        
        # 7. Opcional: publicar para VISP
        if self.publish_for_visp:
            self.publish_to_visp(calibration_pairs)
        
        self.get_logger().info("\n" + "="*50)
        self.get_logger().info("✅ CALIBRACIÓN COMPLETADA")
        self.get_logger().info("="*50)

    def extract_index_from_filename(self, filename):
        """Extrae índice numérico del nombre de archivo"""
        import re
        basename = os.path.basename(filename)
        match = re.search(r'(\d+)', basename)
        if match:
            return int(match.group(1))
        return None

    def run_calibration_methods(self, R_base_to_tool, t_base_to_tool, 
                                 R_board_in_cam, t_board_in_cam):
        """
        Ejecuta diferentes métodos de calibración y compara resultados
        """
        methods = [
            (cv2.CALIB_HAND_EYE_TSAI, "Tsai"),
            (cv2.CALIB_HAND_EYE_PARK, "Park"),
            (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
            (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff"),
            (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis")
        ]
        
        results = []
        
        for method, name in methods:
            try:
                R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                    R_base_to_tool, t_base_to_tool,
                    R_board_in_cam, t_board_in_cam,
                    method=method
                )
                
                # Calcular error
                error = self.compute_calibration_error(
                    R_base_to_tool, t_base_to_tool,
                    R_board_in_cam, t_board_in_cam,
                    R_cam2gripper, t_cam2gripper
                )
                
                results.append({
                    'method': name,
                    'R': R_cam2gripper,
                    't': t_cam2gripper,
                    'error': error
                })
                
                self.get_logger().info(f"   📊 {name:10s}: error = {error:.6f}")
                
            except Exception as e:
                self.get_logger().warn(f"   ⚠️ {name} falló: {e}")
        
        # Ordenar por error
        results.sort(key=lambda x: x['error'])
        
        return results

    def compute_calibration_error(self, R_base_to_tool, t_base_to_tool,
                                    R_board_in_cam, t_board_in_cam,
                                    R_cam2gripper, t_cam2gripper):
        """Calcula error de reproyección"""
        errors = []
        
        for i in range(len(R_base_to_tool)):
            # Construir matrices homogéneas
            H_base_to_tool = np.eye(4)
            H_base_to_tool[:3, :3] = R_base_to_tool[i]
            H_base_to_tool[:3, 3] = t_base_to_tool[i].flatten()
            
            H_board_in_cam = np.eye(4)
            H_board_in_cam[:3, :3] = R_board_in_cam[i]
            H_board_in_cam[:3, 3] = t_board_in_cam[i].flatten()
            
            H_cam2gripper = np.eye(4)
            H_cam2gripper[:3, :3] = R_cam2gripper
            H_cam2gripper[:3, 3] = t_cam2gripper.flatten()
            
            # Para eye-in-hand: H_base_to_tool * H_cam2gripper * H_board_in_cam ≈ constante
            H_pred = H_base_to_tool @ H_cam2gripper @ H_board_in_cam
            error = np.linalg.norm(H_pred[:3, 3] - H_pred[0:3, 0])  # Simplificado
            errors.append(error)
        
        return np.mean(errors)

    def save_calibration_results(self, results, calibration_pairs):
        """
        Guarda los resultados de calibración
        """
        if not results:
            self.get_logger().error("❌ No hay resultados para guardar")
            return
        
        best = results[0]
        
        # Matriz de transformación completa
        T = np.vstack([np.hstack([best['R'], best['t'].reshape(3, 1)]), [0, 0, 0, 1]])
        quat = tf_transformations.quaternion_from_matrix(T)
        
        # Preparar datos
        calibration_data = {
            'calibration_date': datetime.now().isoformat(),
            'eye_in_hand': self.eye_in_hand,
            'best_method': best['method'],
            'error': float(best['error']),
            'num_pairs': len(calibration_pairs),
            'transform_matrix': T.tolist(),
            'rotation_matrix': best['R'].tolist(),
            'translation': best['t'].flatten().tolist(),
            'quaternion': quat.tolist(),
            'charuco_config': {
                'rows': self.rows,
                'cols': self.cols,
                'square_length': self.square_length,
                'marker_length': self.marker_length,
                'dictionary': self.dictionary_name
            },
            'all_methods': [
                {
                    'method': r['method'],
                    'error': float(r['error']),
                    'rotation': r['R'].tolist(),
                    'translation': r['t'].flatten().tolist()
                }
                for r in results
            ],
            'calibration_pairs': [
                {
                    'image': os.path.basename(p['image']),
                    'robot_pose': os.path.basename(p['robot_pose_file']),
                    'num_corners': p['num_corners']
                }
                for p in calibration_pairs
            ]
        }
        
        # Guardar archivo principal
        output_path = os.path.join(self.output_folder, 'hand_eye_calibration.yaml')
        with open(output_path, 'w') as f:
            yaml.dump(calibration_data, f, default_flow_style=False, sort_keys=False)
        
        self.get_logger().info(f"💾 Resultados guardados en: {output_path}")
        
        # Guardar en formato INI para compatibilidad con código anterior
        if best['method'] == "Tsai":  # O el que prefieras
            self.save_ini_format(best, output_path.replace('.yaml', '.ini'))

    def save_ini_format(self, result, output_path):
        """Guarda en formato INI para compatibilidad con código anterior"""
        import configparser
        
        config = configparser.ConfigParser()
        config.optionxform = str
        
        if 'hand_eye_calibration' not in config.sections():
            config.add_section('hand_eye_calibration')
        
        config.set('hand_eye_calibration', 'x', str(result['t'][0, 0]))
        config.set('hand_eye_calibration', 'y', str(result['t'][1, 0]))
        config.set('hand_eye_calibration', 'z', str(result['t'][2, 0]))
        
        # Convertir matriz de rotación a cuaternión
        quat = tf_transformations.quaternion_from_matrix(
            np.vstack([np.hstack([result['R'], result['t'].reshape(3, 1)]), [0, 0, 0, 1]])
        )
        
        config.set('hand_eye_calibration', 'qx', str(quat[0]))
        config.set('hand_eye_calibration', 'qy', str(quat[1]))
        config.set('hand_eye_calibration', 'qz', str(quat[2]))
        config.set('hand_eye_calibration', 'qw', str(quat[3]))
        
        with open(output_path, 'w') as f:
            config.write(f)
        
        self.get_logger().info(f"💾 Formato INI guardado en: {output_path}")

    def publish_to_visp(self, calibration_pairs):
        """
        Opcional: Publica los datos para que VISP pueda procesarlos
        (útil para visualización o debug)
        """
        world_effector_msg = TransformArray()
        camera_object_msg = TransformArray()
        
        world_effector_msg.header = Header()
        world_effector_msg.header.stamp = self.get_clock().now().to_msg()
        world_effector_msg.header.frame_id = 'base_link'
        
        camera_object_msg.header = Header()
        camera_object_msg.header.stamp = self.get_clock().now().to_msg()
        camera_object_msg.header.frame_id = 'calib_camera'
        
        for pair in calibration_pairs:
            # Transform base → tool
            trans = Transform()
            
            # Para eye-in-hand: publicar directamente
            if self.eye_in_hand:
                trans.translation.x = float(pair['t_base_to_tool'][0])
                trans.translation.y = float(pair['t_base_to_tool'][1])
                trans.translation.z = float(pair['t_base_to_tool'][2])
                
                quat = tf_transformations.quaternion_from_matrix(
                    np.vstack([np.hstack([pair['R_base_to_tool'], [[0],[0],[0]]]), [0,0,0,1]])
                )
                trans.rotation.x = quat[0]
                trans.rotation.y = quat[1]
                trans.rotation.z = quat[2]
                trans.rotation.w = quat[3]
                
                world_effector_msg.transforms.append(trans)
            
            # Transform cámara → charuco
            trans_cam = Transform()
            trans_cam.translation.x = float(pair['t_board_in_cam'][0])
            trans_cam.translation.y = float(pair['t_board_in_cam'][1])
            trans_cam.translation.z = float(pair['t_board_in_cam'][2])
            
            quat_cam = tf_transformations.quaternion_from_matrix(
                np.vstack([np.hstack([pair['R_board_in_cam'], [[0],[0],[0]]]), [0,0,0,1]])
            )
            trans_cam.rotation.x = quat_cam[0]
            trans_cam.rotation.y = quat_cam[1]
            trans_cam.rotation.z = quat_cam[2]
            trans_cam.rotation.w = quat_cam[3]
            
            camera_object_msg.transforms.append(trans_cam)
        
        self.world_effector_pub.publish(world_effector_msg)
        self.camera_object_pub.publish(camera_object_msg)
        
        self.get_logger().info(f"📢 Publicados {len(calibration_pairs)} pares para VISP")

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
