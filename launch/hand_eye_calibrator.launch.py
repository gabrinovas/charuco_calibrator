#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        # Arguments
        DeclareLaunchArgument(
            'pictures_folder',
            default_value='/home/drims/calibrations/extrinsic_calibration/pictures',
            description='Folder with images'
        ),
        DeclareLaunchArgument(
            'robot_poses_folder',
            default_value='/home/drims/calibrations/extrinsic_calibration/robot_poses',
            description='Folder with robot poses'
        ),
        DeclareLaunchArgument(
            'output_folder',
            default_value='/home/drims/calibrations/extrinsic_calib_charuco_poses',
            description='Output folder'
        ),
        DeclareLaunchArgument(
            'eye_in_hand',
            default_value='false',
            description='Eye-in-hand or eye-to-hand mode'
        ),
        
        # Offline calibration node
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
