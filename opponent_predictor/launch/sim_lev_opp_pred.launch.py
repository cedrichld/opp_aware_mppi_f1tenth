"""Opponent detector + predictor for the levine_blocked sim.

Uses config/sim/{params,lidar_detector}.yaml, which point at the same
lev_blocked.csv raceline MPPI follows (mppi_bringup sim_lev_blocked.launch.py).
Detects opponents from the ego's /scan and publishes /opponent/odom and
/opponent/predicted_path for MPPI's obstacle avoidance.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    predictor_params_file = LaunchConfiguration('predictor_params_file')
    detector_params_file = LaunchConfiguration('detector_params_file')

    detector = Node(
        package='opponent_predictor',
        executable='opponent_lidar_detector_node',
        name='opponent_lidar_detector',
        output='screen',
        parameters=[detector_params_file],
    )

    predictor = Node(
        package='opponent_predictor',
        executable='opponent_predictor_node',
        name='opponent_predictor',
        output='screen',
        parameters=[predictor_params_file],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'predictor_params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('opponent_predictor'),
                'config',
                'sim',
                'params.yaml',
            ]),
            description='YAML file with opponent predictor parameters.',
        ),
        DeclareLaunchArgument(
            'detector_params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('opponent_predictor'),
                'config',
                'sim',
                'lidar_detector.yaml',
            ]),
            description='YAML file with LiDAR opponent detector parameters.',
        ),
        detector,
        predictor,
    ])
