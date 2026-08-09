"""MPPI against the f1tenth_gym sim on the levine_blocked map.

Sim counterpart of ICRA_main.launch.py with the ICRA files swapped out:
  - trajectory: waypoints/old/lev_testing/lev_blocked.csv
  - wall map:   maps/levine_blocked.yaml
  - params:     params_ICRA1.yaml with is_sim forced true (/ego_racecar/odom)

Pairs with the 4-car sim (four-cars branch of f1tenth_gym_ros, map
levine_blocked) and the pure pursuit opponents:

    ros2 launch f1tenth_gym_ros gym_bridge_launch.py
    ros2 launch pure_pursuit opp_drivers_launch.py
    ros2 launch opponent_predictor sim_lev_opp_pred.launch.py
    ros2 launch mppi_bringup sim_lev_blocked.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    params_file = LaunchConfiguration('params_file')
    drive_topic = LaunchConfiguration('drive_topic')
    wall_map_yaml = LaunchConfiguration('wall_map_yaml')

    # Stop drive while MPPI warms up
    stop_drive = ExecuteProcess(
        cmd=[
            'ros2',
            'topic',
            'pub',
            '--times', '10',
            '--rate', '10',
            '--print', '0',
            '--wait-matching-subscriptions', '0',
            drive_topic,
            'ackermann_msgs/msg/AckermannDriveStamped',
            '{drive: {steering_angle: 0.0, speed: 0.0}}',
        ],
        output='screen',
    )

    pkg_share = get_package_share_directory('mppi_bringup')
    csv_path = os.path.join(pkg_share, 'waypoints', 'old', 'lev_testing', 'lev_blocked.csv')

    mppi_node = Node(
        package='mppi_example',
        executable='mppi_node',
        name='lmppi_node',
        output='screen',
        parameters=[params_file, {
            'is_sim': True,
            'wpt_path': csv_path,
            'wpt_path_absolute': True,
            'wall_cost_map_yaml': wall_map_yaml,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('mppi_bringup'),
                'config',
                'params_ICRA1.yaml',
            ]),
            description='YAML with MPPI ROS2 params (is_sim overridden to true)',
        ),
        DeclareLaunchArgument(
            'drive_topic',
            default_value='/drive',
            description='Ackermann drive topic to stop before starting MPPI',
        ),
        DeclareLaunchArgument(
            'wall_map_yaml',
            default_value=PathJoinSubstitution([
                FindPackageShare('mppi_bringup'),
                'maps',
                'levine_blocked.yaml',
            ]),
            description='Static map yaml used to build the wall-distance cost field',
        ),
        stop_drive,
        TimerAction(
            period=1.6,
            actions=[mppi_node],
        ),
    ])
