#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Ruta al archivo de configuración por defecto
    default_config_file = os.path.join(
        get_package_share_directory('charuco_calibrator'),
        'config',
        'charuco_params.yaml'
    )
    
    return LaunchDescription([
        # Argumentos para calibración ojo-mano
        DeclareLaunchArgument(
            'images_folder',
            default_value='',
            description='Carpeta con las imágenes para calibración ojo-mano'
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config_file,
            description='Archivo de configuración YAML con parámetros del Charuco'
        ),
        DeclareLaunchArgument(
            'camera_intrinsics_file',
            default_value='calibration.yaml',
            description='Archivo YAML con calibración intrínseca de la cámara'
        ),
        DeclareLaunchArgument(
            'robot_poses_file',
            default_value='robot_poses.txt',
            description='Archivo con las poses del robot'
        ),
        DeclareLaunchArgument(
            'hand_eye_output',
            default_value='hand_eye_calibration.yaml',
            description='Archivo de salida para la calibración ojo-mano'
        ),
        
        # Nodo de calibración ojo-mano
        Node(
            package='charuco_calibrator',
            executable='charuco_hand_eye',
            name='charuco_hand_eye_calibrator',
            output='screen',
            parameters=[{
                'images_folder': LaunchConfiguration('images_folder'),
                'config_file': LaunchConfiguration('config_file'),
                'camera_intrinsics_file': LaunchConfiguration('camera_intrinsics_file'),
                'robot_poses_file': LaunchConfiguration('robot_poses_file'),
                'hand_eye_output': LaunchConfiguration('hand_eye_output'),
            }],
            emulate_tty=True,
        )
    ])