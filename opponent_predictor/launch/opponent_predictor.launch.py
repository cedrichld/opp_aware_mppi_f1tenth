from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration('params_file')

    predictor = Node(
        package='opponent_predictor',
        executable='opponent_predictor_node',
        name='opponent_predictor',
        output='screen',
        parameters=[params_file],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('opponent_predictor'),
                'config',
                'params.yaml',
            ]),
            description='YAML file with opponent predictor parameters.',
        ),
        predictor,
    ])
