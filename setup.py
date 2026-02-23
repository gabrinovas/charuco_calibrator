from setuptools import setup
import os
from glob import glob

package_name = 'charuco_calibrator'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Gabriel Novas',
    maintainer_email='gabriel.novas@aimen.es',
    description='Charuco detector for camera calibration and hand-eye calibration',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'charuco_intrinsic = charuco_calibrator.charuco_intrinsic:main',
            'charuco_hand_eye = charuco_calibrator.charuco_hand_eye:main',
            'capture_for_calibration = charuco_calibrator.capture_for_calibration:main',
        ],
    },
)