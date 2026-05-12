import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'diff_bot_3'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='johan',
    maintainer_email='johan@todo.com',
    description='Robot diferencial + montacargas autonomo (RPi5 maestro, ESP32 esclavo)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_core        = diff_bot_3.robot_core:main',
            'robot_teleop      = diff_bot_3.robot_teleop:main',
            'robot_interface   = diff_bot_3.robot_interface:main',
            'robot_player      = diff_bot_3.robot_player:main',
            'vision_node       = diff_bot_3.vision_node:main',
            'forklift_manager  = diff_bot_3.forklift_manager:main',
            'esp32_bridge      = diff_bot_3.esp32_bridge:main',
            'cmd_vel_mux       = diff_bot_3.cmd_vel_mux:main',
        ],
    },
)
