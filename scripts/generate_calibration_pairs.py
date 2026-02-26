#!/usr/bin/env python3
"""
Script para generar pares de calibración a partir de imágenes y poses guardadas.
Ejecutar después de tener todas las imágenes y poses del robot.
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
    
    # Crear carpetas de salida
    PAIRS_FOLDER = os.path.join(OUTPUT_FOLDER, "pairs")
    DETECTIONS_FOLDER = os.path.join(OUTPUT_FOLDER, "detections")
    os.makedirs(PAIRS_FOLDER, exist_ok=True)
    os.makedirs(DETECTIONS_FOLDER, exist_ok=True)
    
    # Cargar configuración del tablero
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    calib_params = config.get('charuco_calibrator', {}).get('ros__parameters', {})
    
    rows = calib_params.get('charuco_rows', 14)
    cols = calib_params.get('charuco_cols', 10)
    square_length = calib_params.get('square_length', 0.020)
    marker_length = calib_params.get('marker_length', 0.015)
    dictionary_name = calib_params.get('dictionary', 'DICT_4X4_100')
    
    # Configurar detector
    dictionary_map = {
        'DICT_4X4_100': aruco.DICT_4X4_100,
        'DICT_4X4_250': aruco.DICT_4X4_250,
        # ... otros diccionarios
    }
    
    dict_id = dictionary_map.get(dictionary_name, aruco.DICT_4X4_100)
    aruco_dict = aruco.getPredefinedDictionary(dict_id)
    
    board = aruco.CharucoBoard(
        (cols, rows),
        square_length,
        marker_length,
        aruco_dict
    )
    
    # Cargar intrínsecos de cámara (asumiendo que están en pictures_folder)
    intrinsics_file = os.path.join(PICTURES_FOLDER, 'calibration.yaml')
    if os.path.exists(intrinsics_file):
        with open(intrinsics_file, 'r') as f:
            intrinsics = yaml.safe_load(f)
        camera_matrix = np.array(intrinsics['camera_matrix'])
        dist_coeffs = np.array(intrinsics['distortion_coefficients'])
    else:
        print("⚠️ No se encontró archivo de intrínsecos, usando valores por defecto")
        camera_matrix = np.eye(3)
        dist_coeffs = np.zeros(5)
    
    # Buscar imágenes
    image_paths = sorted(glob.glob(os.path.join(PICTURES_FOLDER, '*.jpg')) +
                         glob.glob(os.path.join(PICTURES_FOLDER, '*.png')))
    
    # Buscar poses del robot
    pose_files = sorted(glob.glob(os.path.join(ROBOT_POSES_FOLDER, 'pose_*.yaml')))
    
    print(f"📸 Imágenes encontradas: {len(image_paths)}")
    print(f"🤖 Poses encontradas: {len(pose_files)}")
    
    # Procesar cada par
    valid_pairs = 0
    
    for i, (image_path, pose_file) in enumerate(zip(image_paths, pose_files)):
        print(f"\n📌 Procesando par {i+1}:")
        print(f"   Imagen: {os.path.basename(image_path)}")
        print(f"   Pose: {os.path.basename(pose_file)}")
        
        # Leer pose del robot
        with open(pose_file, 'r') as f:
            pose_data = yaml.safe_load(f)
        
        # Leer imagen
        img = cv2.imread(image_path)
        if img is None:
            print(f"   ❌ No se pudo leer la imagen")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detectar marcadores
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
        
        if ids is None or len(ids) < 4:
            print(f"   ❌ No se detectaron suficientes marcadores")
            continue
        
        # Interpolar esquinas Charuco
        ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        
        if charuco_corners is None or len(charuco_corners) < 10:
            print(f"   ❌ No hay suficientes esquinas Charuco")
            continue
        
        # Estimar pose
        ret, rvec, tvec = aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board,
            camera_matrix, dist_coeffs, None, None
        )
        
        if not ret:
            print(f"   ❌ No se pudo estimar la pose")
            continue
        
        # Convertir a matriz de rotación
        R, _ = cv2.Rodrigues(rvec)
        
        # Obtener cuaternión
        T = np.vstack([np.hstack([R, tvec.reshape(3, 1)]), [0, 0, 0, 1]])
        quat = tf_transformations.quaternion_from_matrix(T)
        
        # Guardar detección individual
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
        
        # Guardar par completo
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
        
        # Visualización
        img_viz = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
        img_viz = cv2.aruco.drawDetectedCornersCharuco(img_viz, charuco_corners, charuco_ids)
        cv2.drawFrameAxes(img_viz, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
        
        viz_path = os.path.join(OUTPUT_FOLDER, f"viz_{i+1:03d}.jpg")
        cv2.imwrite(viz_path, img_viz)
        
        valid_pairs += 1
        print(f"   ✅ Par válido #{valid_pairs} ({len(charuco_corners)} esquinas)")
    
    # Guardar resumen
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
    print(f"✅ Procesamiento completado")
    print(f"📊 Pares válidos: {valid_pairs}/{len(image_paths)}")
    print(f"📁 Detecciones: {DETECTIONS_FOLDER}")
    print(f"📁 Pares: {PAIRS_FOLDER}")
    print(f"📄 Resumen: {summary_file}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
