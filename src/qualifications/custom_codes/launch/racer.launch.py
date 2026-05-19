from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    autodrive_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("autodrive_roboracer"),
                "launch",
                "bringup_headless.launch.py",
            )
        )
    )

    racer_node = Node(
        package="custom_codes",
        executable="racer_node",
        name="racer_node",
        output="screen",
    )

    return LaunchDescription([
        autodrive_launch,
        racer_node,
    ])