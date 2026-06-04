from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    params_file = LaunchConfiguration('params_file')
    drive_topic = LaunchConfiguration('drive_topic')
    wall_map_yaml = LaunchConfiguration('wall_map_yaml')

    # Bridge dual-region launch toggle (the only knob in the launch file —
    # all bubble/centre tuning lives in region_config_file yaml).
    bridge_enabled = LaunchConfiguration('bridge_enabled')
    region_config_file = LaunchConfiguration('region_config_file')
    # Paths to the OVER raceline + OVER wall map. Resolved via FindPackageShare
    # at runtime so the same launch file works on dev (cedric) and Jetson (nvidia)
    # without hardcoded absolute paths.
    over_wpt_path = LaunchConfiguration('over_wpt_path')
    over_wall_yaml = LaunchConfiguration('over_wall_yaml')

    # Stop drive while MPPI warms up
    stop_drive = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--times', '10', '--rate', '10',
            '--print', '0', '--wait-matching-subscriptions', '0',
            drive_topic,
            'ackermann_msgs/msg/AckermannDriveStamped',
            '{drive: {steering_angle: 0.0, speed: 0.0}}',
        ],
        output='screen',
    )

    pkg_share = get_package_share_directory('mppi_bringup')
    csv_path = os.path.join(pkg_share, 'waypoints', 'TODO.csv')

    # MPPI gets its own bridge params (over_wpt_path, over_wall_cost_map_yaml,
    # region_transition_silence_s) from params_ICRA1.yaml — NOT from launch.
    mppi_node = Node(
        package='mppi_example',
        executable='mppi_node',
        name='lmppi_node',
        output='screen',
        parameters=[params_file, {
            'wpt_path': csv_path,
            'wpt_path_absolute': True,
            'wall_cost_map_yaml': wall_map_yaml,
            # Same #injected-by-launch pattern as wall_cost_map_yaml: yaml has
            # empty strings, launch fills the machine-correct absolute paths.
            'over_wpt_path': over_wpt_path,
            'over_wall_cost_map_yaml': over_wall_yaml,
        }],
    )

    # region_manager params come entirely from region_config_file yaml.
    # Launches only when bridge_enabled:=true.
    region_manager = Node(
        package='region_manager',
        executable='region_manager',
        name='region_manager',
        output='screen',
        condition=IfCondition(bridge_enabled),
        parameters=[region_config_file],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('mppi_bringup'),
                'config',
                'params_ICRA1.yaml',
            ]),
            description='YAML with MPPI ROS2 params for actual car',
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
                'ICRA_1.yaml',
            ]),
            description='Static map yaml used to build the wall-distance cost field (UNDER region)',
        ),
        # ── Bridge mode ──
        DeclareLaunchArgument('bridge_enabled', default_value='true',
                              description='Enable dual-region (under/over) bridge mode and launch region_manager'),
        DeclareLaunchArgument(
            'region_config_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('mppi_bringup'),
                'config',
                'region_ICRA_Masters.yaml',
            ]),
            description='Yaml with region_manager params (bubble centres, radii, pose topic).',
        ),
        # Portable absolute paths for the OVER files (machine-independent).
        # Replace ICRA_Masters/over.csv and ICRA_Masters/over_map.yaml here only
        # if you reorganise the files in mppi_bringup/{waypoints,maps}.
        DeclareLaunchArgument(
            'over_wpt_path',
            default_value=PathJoinSubstitution([
                FindPackageShare('mppi_bringup'),
                'waypoints',
                'ICRA_Masters', 'over.csv',
            ]),
            description='Absolute path to OVER-bridge raceline CSV (resolved via FindPackageShare).',
        ),
        DeclareLaunchArgument(
            'over_wall_yaml',
            default_value=PathJoinSubstitution([
                FindPackageShare('mppi_bringup'),
                'maps',
                'ICRA_Masters', 'over_map.yaml',
            ]),
            description='Absolute path to OVER-bridge wall-cost map yaml (resolved via FindPackageShare).',
        ),

        stop_drive,
        TimerAction(period=1.6, actions=[mppi_node]),
        region_manager,
    ])
