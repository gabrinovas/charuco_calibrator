#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    default_config_file = os.path.join(
        get_package_share_directory('charuco_calibrator'),
        'config',
        'charuco_params.yaml'
    )
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'images_folder',
            default_value='',
            description='Folder with images for calibration'
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config_file,
            description='YAML configuration file'
        ),
        DeclareLaunchArgument(
            'output_file',
            default_value='calibration.yaml',
            description='Name of the output file'
        ),
        DeclareLaunchArgument(
            'show_preview',
            default_value='false',
            description='Show preview'
        ),
        
        Node(
            package='charuco_calibrator',
            executable='charuco_intrinsic',
            name='charuco_intrinsic_calibrator',
            output='screen',
            parameters=[{
                'images_folder': LaunchConfiguration('images_folder'),
                'config_file': LaunchConfiguration('config_file'),
                'output_file': LaunchConfiguration('output_file'),
            }],
            emulate_tty=True,
        )
    ])
