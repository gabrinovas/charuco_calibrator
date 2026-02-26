#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        # Argumentos
        DeclareLaunchArgument(
            'pictures_folder',
            default_value='/home/drims/drims_ws/calibrations/extrinsic_calibration/pictures',
            description='Carpeta con imágenes'
        ),
        DeclareLaunchArgument(
            'robot_poses_folder',
            default_value='/home/drims/drims_ws/calibrations/extrinsic_calibration/robot_poses',
            description='Carpeta con poses del robot'
        ),
        DeclareLaunchArgument(
            'output_folder',
            default_value='/home/drims/drims_ws/calibrations/extrinsic_calib_charuco_poses',
            description='Carpeta de salida'
        ),
        DeclareLaunchArgument(
            'eye_in_hand',
            default_value='false',
            description='Modo eye-in-hand o eye-to-hand'
        ),
        
        # Nodo de calibración offline
        Node(
            package='charuco_calibrator',
            executable='charuco_hand_eye_offline',
            name='charuco_offline_calibration',
            output='screen',
            parameters=[{
                'pictures_folder': LaunchConfiguration('pictures_folder'),
                'robot_poses_folder': LaunchConfiguration('robot_poses_folder'),
                'output_folder': LaunchConfiguration('output_folder'),
                'eye_in_hand': LaunchConfiguration('eye_in_hand'),
                'publish_for_visp': False,
            }],
            arguments=['--ros-args', '--log-level', 'info'],
        ),
    ])
