#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # 1. Declare a launch argument use_sim_time and make it defaults to "true".
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription(
        [
            use_sim_time_arg,

            # 3. Launch an existing launch file rviz.launch.py package asl_tb3_sim
            IncludeLaunchDescription(
                PathJoinSubstitution(
                    [FindPackageShare("asl_tb3_sim"), "launch", "rviz.launch.py"]
                ),
                launch_arguments={
                    # (a) Set config to the path of your default.rviz.
                    "config": PathJoinSubstitution(
                        [
                            FindPackageShare("autonomy_repo"),
                            "rviz",
                            "default.rviz",
                        ]
                    ),
                    # (b) Set use_sim_time to the launch argument defined above.
                    "use_sim_time": use_sim_time,
                }.items(),
            ),

            # 2. Launch the following nodes
            # (a) Node rviz_goal_relay.py from package asl_tb3_lib. Set parameter output_channel to /cmd_nav.
            Node(
                executable="rviz_goal_relay.py",
                package="asl_tb3_lib",
                parameters=[
                    {"output_channel": "/cmd_nav"},
                ],
            ),

            # (b) Node state_publisher.py from package asl_tb3_lib.
            Node(
                executable="state_publisher.py",
                package="asl_tb3_lib",
                parameters=[
                    {"use_sim_time": use_sim_time}, # It's good practice to pass use_sim_time to all nodes
                ],
            ),

            # (c) Node navigator.py from package autonomy_repo (This is your navigator node!).
            # Set parameter use_sim_time to the launch argument defined above.
            Node(
                executable="navigator.py",
                package="autonomy_repo",
                parameters=[
                    {"use_sim_time": use_sim_time},
                    # You might want to add other parameters for your navigator node here,
                    # e.g., kpx, kpy, kdx, kdy, V_PREV_THRESH, v_desired, spline_alpha, kp_heading
                    # Example: {"kpx": 1.0, "kpy": 1.0, "kdx": 1.0, "kdy": 1.0},
                ],
            ),
        ]
    )