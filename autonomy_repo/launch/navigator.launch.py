#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time", default_value="true", description="Use simulation clock if true"
            ),
            IncludeLaunchDescription(
                PathJoinSubstitution(
                    [FindPackageShare("asl_tb3_sim"), "launch", "rviz.launch.py"]
                ),
                launch_arguments={
                    "config": PathJoinSubstitution(
                        [FindPackageShare("autonomy_repo"), "rviz", "default.rviz"]
                    ),
                    "use_sim_time": use_sim_time,
                }.items(),
            ),
            Node(
                package="asl_tb3_lib",
                executable="rviz_goal_relay.py",
                name="rviz_goal_relay",
                parameters=[{"output_channel": "/cmd_nav"}],
            ),
            Node(
                package="asl_tb3_lib",
                executable="state_publisher.py",
                name="state_publisher",
            ),
            Node(
                package="autonomy_repo",
                executable="navigator.py",
                name="navigator",
                parameters=[{"use_sim_time": use_sim_time}],
                output="screen",  # Optional: Display node's output in the terminal
            ),
        ]
    )