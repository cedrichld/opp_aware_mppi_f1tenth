import os
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path as NavPath
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from std_msgs.msg import Float32, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from raceline_msgs.srv import UpdateRaceline

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None

try:
    from .utils.ros_np_multiarray import to_multiarray_f32
    from .infer_env import InferEnv
    from .mppi_tracking import MPPI
    from .utils import utils
    from .utils.jax_utils import numpify
    from .utils import jax_utils
    from .utils.Track import Track
except ImportError:
    from utils.ros_np_multiarray import to_multiarray_f32
    from infer_env import InferEnv
    from mppi_tracking import MPPI
    import utils.utils as utils
    from utils.jax_utils import numpify
    import utils.jax_utils as jax_utils
    from utils.Track import Track
jax.config.update("jax_compilation_cache_dir",
                  os.environ.get("JAX_CACHE_DIR", os.path.expanduser("~/jax_cache")))


## This is a demosntration of how to use the MPPI planner with the Roboracer
## Zirui Zang 2025/04/07

class MPPI_Node(Node):
    DEBUG_SCALAR_TOPICS = {
        'reward_total_sum': '/mppi/debug/reward_total_sum',
        'reward_total_mean': '/mppi/debug/reward_total_mean',
        'reward_xy_sum': '/mppi/debug/reward_xy_sum',
        'reward_velocity_sum': '/mppi/debug/reward_velocity_sum',
        'reward_yaw_sum': '/mppi/debug/reward_yaw_sum',
        'cost_total_sum': '/mppi/debug/cost_total_sum',
        'cost_total_mean': '/mppi/debug/cost_total_mean',
        'cost_invalid_sum': '/mppi/debug/cost_invalid_sum',
        'cost_wall_sum': '/mppi/debug/cost_wall_sum',
        'cost_slip_sum': '/mppi/debug/cost_slip_sum',
        'cost_latacc_sum': '/mppi/debug/cost_latacc_sum',
        'cost_steer_sat_sum': '/mppi/debug/cost_steer_sat_sum',
        'cost_opponent_sum': '/mppi/debug/cost_opponent_sum',
        'min_wall_dist': '/mppi/debug/min_wall_dist',
        'min_opponent_dist': '/mppi/debug/min_opponent_dist',
        'opponent_path_age': '/mppi/debug/opponent_path_age',
        'opponent_active': '/mppi/debug/opponent_active',
        'max_beta': '/mppi/debug/max_beta',
        'max_latacc': '/mppi/debug/max_latacc',
        'max_abs_steer': '/mppi/debug/max_abs_steer',
        'invalid_steps': '/mppi/debug/invalid_steps',
    }

    def __init__(self):
        super().__init__('mppi_node')
        self.config = utils.ConfigYAML()
        self.config_path = self.default_config_path()
        self.config.load_file(str(self.config_path))
        self.config.norm_params = np.array(self.config.norm_params, dtype=float).T
        self.ensure_config_defaults()
        self.declare_ros_params()
        self.get_params(startup=True)

        if self.config.random_seed is None:
            self.config.random_seed = np.random.randint(0, 1e6)
        jrng = jax_utils.oneLineJaxRNG(self.config.random_seed)

        if self.config.wpt_path_absolute and self.config.wpt_path:
            self.get_logger().info(f'Loading raceline directly from {self.config.wpt_path}')
            track, self.config = Track.load_map_from_csv(self.config.wpt_path, self.config)
        else:
            self.resolve_map_dir()
            map_info = np.genfromtxt(self.config.map_dir + 'map_info.txt', delimiter='|', dtype='str')
            track, self.config = Track.load_map(self.config.map_dir, map_info, self.config.map_ind, self.config)
        # track.waypoints[:, 3] += 0.5 * np.pi
        self.infer_env = InferEnv(track, self.config, DT=self.config.sim_time_step)
        self.mppi = MPPI(
            self.config,
            self.infer_env,
            jrng,
            temperature=self.config.temperature,
            damping=self.config.damping,
        )
        self.get_params()

        # Do a dummy call on the MPPI to initialize the variables
        state_c_0 = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.control = np.asarray([0.0, 0.0])
        reference_traj, waypoint_ind = self.infer_env.get_refernece_traj(state_c_0.copy(), self.config.ref_vel, self.config.n_steps)
        
        self.opponent_xy_horizon = np.zeros((self.config.n_steps, 2), dtype=np.float32)
        self.opponent_path_time = None
        self.mppi.update(
            jnp.asarray(state_c_0),
            jnp.asarray(reference_traj),
            jnp.asarray(self.opponent_xy_horizon),
            False,
        )
        self.get_logger().info('MPPI initialized')
        self.hz = []
        self.last_speed_command_time = None
        self.prev_pose_time = None
        self.prev_pose_xy = None
        self.prev_pose_yaw = None
        self.state_est_vx = None
        self.state_est_vy = 0.0
        self.state_est_wz = 0.0
        self.state_est_method_enabled = bool(self.config.use_pose_delta_state_estimate)

        
        qos = rclpy.qos.QoSProfile(history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
                                   depth=1,
                                   reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
                                   durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE)
        # create subscribers
        if self.config.is_sim:
            self.pose_sub = self.create_subscription(Odometry, "/ego_racecar/odom", self.pose_callback, qos)
        else:
            self.pose_sub = self.create_subscription(Odometry, "/pf/pose/odom", self.pose_callback, qos)
        self.opponent_path_sub = self.create_subscription(
            NavPath,
            self.config.opponent_path_topic,
            self.opponent_path_callback,
            qos,
        )
        # publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", qos)
        self.reference_pub = self.create_publisher(Float32MultiArray, "/reference_arr", qos)
        self.opt_traj_pub = self.create_publisher(Float32MultiArray, "/opt_traj_arr", qos)
        self.reference_marker_pub = self.create_publisher(MarkerArray, "/mppi/reference", qos)
        self.opt_traj_marker_pub = self.create_publisher(MarkerArray, "/mppi/optimal_trajectory", qos)
        self.sampled_traj_marker_pub = self.create_publisher(MarkerArray, "/mppi/sampled_trajectories", qos)
        self.speed_debug_pub = self.create_publisher(Float32MultiArray, "/mppi/speed_debug", qos)
        self.reward_debug_pubs = {
            key: self.create_publisher(Float32, topic, qos)
            for key, topic in self.DEBUG_SCALAR_TOPICS.items()
        }

        self.update_raceline_srv = self.create_service(
            UpdateRaceline, '/mppi/update_raceline', self.update_raceline_callback
        )
        self.get_logger().info('Raceline update service ready at /mppi/update_raceline')

    def update_raceline_callback(self, request, response):
        try:
            if request.format != 'mppi':
                response.success = False
                response.message = f"expected format='mppi', got '{request.format}'"
                return response
            if request.cols != 9:
                response.success = False
                response.message = f"expected cols=9, got {request.cols}"
                return response
            expected_rows = int(self.infer_env.waypoints.shape[0])
            if int(request.rows) != expected_rows:
                response.success = False
                response.message = (
                    f"row count mismatch: got {request.rows}, expected {expected_rows} "
                    f"(changing N would trigger JAX re-JIT; restart the node to change size)"
                )
                return response
            data = np.asarray(request.data, dtype=np.float64)
            if data.size != request.rows * request.cols:
                response.success = False
                response.message = (
                    f"data length {data.size} != rows*cols {request.rows * request.cols}"
                )
                return response
            waypoints = data.reshape(int(request.rows), int(request.cols))
            self.infer_env.update_waypoints(waypoints)
            response.success = True
            response.message = f"hot-swapped {waypoints.shape[0]} waypoints"
            self.get_logger().info(response.message)
            return response
        except Exception as exc:
            response.success = False
            response.message = f"update failed: {exc}"
            self.get_logger().error(response.message)
            return response

    def default_config_path(self):
        if get_package_share_directory is not None:
            try:
                return Path(get_package_share_directory('mppi_example')) / 'config.yaml'
            except Exception:
                pass
        return Path(__file__).resolve().parent / 'config.yaml'

    def ensure_config_defaults(self):
        defaults = {
            'wpt_path_absolute': False,
            'wpt_path': '',
            'temperature': 0.01,
            'damping': 0.001,
            'xy_reward_weight': 1.0,
            'velocity_reward_weight': 0.0,
            'yaw_reward_weight': 0.0,
            'use_waypoint_speed_profile': False,
            'speed_profile_scale': 1.0,
            'speed_profile_min_speed': 0.0,
            'speed_profile_max_speed': 20.0,
            'speed_profile_lookahead_steps': 0,
            'speed_profile_iterations': 1,
            'use_speed_profile_drive_speed': False,
            'speed_profile_drive_blend': 0.0,
            'speed_profile_drive_lookahead_steps': 1,
            'speed_profile_drive_use_min_lookahead': False,
            'speed_profile_drive_max_accel': 0.0,
            'speed_profile_drive_max_decel': 0.0,
            'startup_speed': float(self.config.init_vel) * 2.0,
            'use_pose_delta_state_estimate': False,
            'min_speed': 0.0,
            'max_speed': 20.0,
            'max_steering_angle': 0.4189,
            'publish_markers': True,
            'marker_frame_id': 'map',
            'reference_line_width': 0.06,
            'optimal_line_width': 0.08,
            'sampled_line_width': 0.025,
            'sampled_trajectory_count': 64,
            'sampled_trajectory_alpha': 0.18,
            'wall_cost_enabled': False,
            'wall_cost_weight': 0.0,
            'wall_cost_margin': 0.3,
            'wall_cost_power': 2.0,
            'wall_cost_map_yaml': '',
            'opponent_path_topic': '/opponent/predicted_path',
            'opponent_cost_enabled': False,
            'opponent_cost_weight': 0.0,
            'opponent_cost_radius': 0.8,
            'opponent_cost_power': 2.0,
            'opponent_cost_discount': 1.0,
            'opponent_path_timeout': 0.5,
            'slip_cost_enabled': False,
            'slip_cost_weight': 0.0,
            'slip_cost_beta_safe': 0.2,
            'latacc_cost_enabled': False,
            'latacc_cost_weight': 0.0,
            'latacc_cost_safe': 8.0,
            'steer_sat_cost_enabled': False,
            'steer_sat_cost_weight': 0.0,
            'steer_sat_soft_ratio': 0.85,

        }
        for key, value in defaults.items():
            if not hasattr(self.config, key):
                setattr(self.config, key, value)

    def declare_ros_params(self):
        steer_vel_scale = float(self.config.norm_params[0, 0] / 2.0)
        accel_scale = float(self.config.norm_params[0, 1] / 2.0)
        random_seed = -1 if self.config.random_seed is None else int(self.config.random_seed)

        startup_params = {
            'is_sim': bool(self.config.is_sim),
            'wpt_path_absolute': bool(self.config.wpt_path_absolute),
            'wpt_path': str(self.config.wpt_path),
            'wall_cost_map_yaml': str(self.config.wall_cost_map_yaml),
            'opponent_path_topic': str(self.config.opponent_path_topic),
            'map_dir': str(self.config.map_dir),
            'map_ind': int(self.config.map_ind),
            'state_predictor': str(self.config.state_predictor),
            'n_samples': int(self.config.n_samples),
            'n_steps': int(self.config.n_steps),
            'sim_time_step': float(self.config.sim_time_step),
            'random_seed': random_seed,
            'render': bool(self.config.render),
        }
        live_params = {
            'temperature': float(self.config.temperature),
            'damping': float(self.config.damping),
            'ref_vel': float(self.config.ref_vel),
            'init_vel': float(self.config.init_vel),
            'startup_speed': float(self.config.startup_speed),
            'use_pose_delta_state_estimate': bool(self.config.use_pose_delta_state_estimate),
            'friction': float(self.config.friction),
            'n_iterations': int(self.config.n_iterations),
            'control_sample_std_steer': float(self.config.control_sample_std[0]),
            'control_sample_std_accel': float(self.config.control_sample_std[1]),
            'steer_vel_scale': steer_vel_scale,
            'accel_scale': accel_scale,
            'xy_reward_weight': float(self.config.xy_reward_weight),
            'velocity_reward_weight': float(self.config.velocity_reward_weight),
            'yaw_reward_weight': float(self.config.yaw_reward_weight),
            'wall_cost_enabled': bool(self.config.wall_cost_enabled),
            'wall_cost_weight': float(self.config.wall_cost_weight),
            'wall_cost_margin': float(self.config.wall_cost_margin),
            'wall_cost_power': float(self.config.wall_cost_power),
            'opponent_cost_enabled': bool(self.config.opponent_cost_enabled),
            'opponent_cost_weight': float(self.config.opponent_cost_weight),
            'opponent_cost_radius': float(self.config.opponent_cost_radius),
            'opponent_cost_power': float(self.config.opponent_cost_power),
            'opponent_cost_discount': float(self.config.opponent_cost_discount),
            'opponent_path_timeout': float(self.config.opponent_path_timeout),
            'slip_cost_enabled': bool(self.config.slip_cost_enabled),
            'slip_cost_weight': float(self.config.slip_cost_weight),
            'slip_cost_beta_safe': float(self.config.slip_cost_beta_safe),
            'latacc_cost_enabled': bool(self.config.latacc_cost_enabled),
            'latacc_cost_weight': float(self.config.latacc_cost_weight),
            'latacc_cost_safe': float(self.config.latacc_cost_safe),
            'steer_sat_cost_enabled': bool(self.config.steer_sat_cost_enabled),
            'steer_sat_cost_weight': float(self.config.steer_sat_cost_weight),
            'steer_sat_soft_ratio': float(self.config.steer_sat_soft_ratio),
            'use_waypoint_speed_profile': bool(self.config.use_waypoint_speed_profile),
            'speed_profile_scale': float(self.config.speed_profile_scale),
            'speed_profile_min_speed': float(self.config.speed_profile_min_speed),
            'speed_profile_max_speed': float(self.config.speed_profile_max_speed),
            'speed_profile_lookahead_steps': int(self.config.speed_profile_lookahead_steps),
            'speed_profile_iterations': int(self.config.speed_profile_iterations),
            'use_speed_profile_drive_speed': bool(self.config.use_speed_profile_drive_speed),
            'speed_profile_drive_blend': float(self.config.speed_profile_drive_blend),
            'speed_profile_drive_lookahead_steps': int(self.config.speed_profile_drive_lookahead_steps),
            'speed_profile_drive_use_min_lookahead': bool(self.config.speed_profile_drive_use_min_lookahead),
            'speed_profile_drive_max_accel': float(self.config.speed_profile_drive_max_accel),
            'speed_profile_drive_max_decel': float(self.config.speed_profile_drive_max_decel),
            'min_speed': float(self.config.min_speed),
            'max_speed': float(self.config.max_speed),
            'max_steering_angle': float(self.config.max_steering_angle),
            'publish_markers': bool(self.config.publish_markers),
            'marker_frame_id': str(self.config.marker_frame_id),
            'reference_line_width': float(self.config.reference_line_width),
            'optimal_line_width': float(self.config.optimal_line_width),
            'sampled_line_width': float(self.config.sampled_line_width),
            'sampled_trajectory_count': int(self.config.sampled_trajectory_count),
            'sampled_trajectory_alpha': float(self.config.sampled_trajectory_alpha),
        }

        for name, value in {**startup_params, **live_params}.items():
            self.declare_parameter(name, value)

    def get_params(self, startup=False):
        if startup:
            self.config.is_sim = bool(self.get_parameter('is_sim').value)
            self.config.wpt_path_absolute = bool(self.get_parameter('wpt_path_absolute').value)
            self.config.wpt_path = str(self.get_parameter('wpt_path').value)
            self.config.wall_cost_map_yaml = str(self.get_parameter('wall_cost_map_yaml').value)
            self.config.opponent_path_topic = str(self.get_parameter('opponent_path_topic').value)
            self.config.map_dir = str(self.get_parameter('map_dir').value)
            self.config.map_ind = int(self.get_parameter('map_ind').value)
            self.config.state_predictor = str(self.get_parameter('state_predictor').value)
            self.config.n_samples = int(self.get_parameter('n_samples').value)
            self.config.n_steps = int(self.get_parameter('n_steps').value)
            self.config.sim_time_step = float(self.get_parameter('sim_time_step').value)
            random_seed = int(self.get_parameter('random_seed').value)
            self.config.random_seed = None if random_seed < 0 else random_seed
            self.config.render = bool(self.get_parameter('render').value)

        self.config.temperature = max(1e-6, float(self.get_parameter('temperature').value))
        self.config.damping = max(1e-9, float(self.get_parameter('damping').value))
        self.config.ref_vel = float(self.get_parameter('ref_vel').value)
        self.config.init_vel = float(self.get_parameter('init_vel').value)
        self.config.startup_speed = float(self.get_parameter('startup_speed').value)
        self.config.use_pose_delta_state_estimate = bool(
            self.get_parameter('use_pose_delta_state_estimate').value
        )
        self.config.friction = float(self.get_parameter('friction').value)
        self.config.n_iterations = max(1, int(self.get_parameter('n_iterations').value))
        self.config.control_sample_std = [
            float(self.get_parameter('control_sample_std_steer').value),
            float(self.get_parameter('control_sample_std_accel').value),
        ]
        steer_vel_scale = abs(float(self.get_parameter('steer_vel_scale').value))
        accel_scale = abs(float(self.get_parameter('accel_scale').value))
        self.config.norm_params = np.array([
            [2.0 * steer_vel_scale, 2.0 * accel_scale],
            [-2.0 * steer_vel_scale, -2.0 * accel_scale],
        ], dtype=float)
        self.config.xy_reward_weight = float(self.get_parameter('xy_reward_weight').value)
        self.config.velocity_reward_weight = float(self.get_parameter('velocity_reward_weight').value)
        self.config.yaw_reward_weight = float(self.get_parameter('yaw_reward_weight').value)
        self.config.wall_cost_enabled = bool(self.get_parameter('wall_cost_enabled').value)
        self.config.wall_cost_weight = max(
            0.0,
            float(self.get_parameter('wall_cost_weight').value),
        )
        self.config.wall_cost_margin = max(
            0.0,
            float(self.get_parameter('wall_cost_margin').value),
        )
        self.config.wall_cost_power = max(
            1.0,
            float(self.get_parameter('wall_cost_power').value),
        )
        self.config.opponent_cost_enabled = bool(self.get_parameter('opponent_cost_enabled').value)
        self.config.opponent_cost_weight = max(
            0.0,
            float(self.get_parameter('opponent_cost_weight').value),
        )
        self.config.opponent_cost_radius = max(
            0.0,
            float(self.get_parameter('opponent_cost_radius').value),
        )
        self.config.opponent_cost_power = max(
            1.0,
            float(self.get_parameter('opponent_cost_power').value),
        )
        self.config.opponent_cost_discount = float(np.clip(
            float(self.get_parameter('opponent_cost_discount').value),
            0.0,
            1.0,
        ))
        self.config.opponent_path_timeout = max(
            0.0,
            float(self.get_parameter('opponent_path_timeout').value),
        )
        self.config.slip_cost_enabled = bool(self.get_parameter('slip_cost_enabled').value)
        self.config.slip_cost_weight = max(
            0.0,
            float(self.get_parameter('slip_cost_weight').value),
        )
        self.config.slip_cost_beta_safe = max(
            0.0,
            float(self.get_parameter('slip_cost_beta_safe').value),
        )
        self.config.latacc_cost_enabled = bool(self.get_parameter('latacc_cost_enabled').value)
        self.config.latacc_cost_weight = max(
            0.0,
            float(self.get_parameter('latacc_cost_weight').value),
        )
        self.config.latacc_cost_safe = max(
            0.0,
            float(self.get_parameter('latacc_cost_safe').value),
        )
        self.config.steer_sat_cost_enabled = bool(self.get_parameter('steer_sat_cost_enabled').value)
        self.config.steer_sat_cost_weight = max(
            0.0,
            float(self.get_parameter('steer_sat_cost_weight').value),
        )
        self.config.steer_sat_soft_ratio = float(np.clip(
            float(self.get_parameter('steer_sat_soft_ratio').value),
            0.0,
            1.0,
        ))
        self.config.use_waypoint_speed_profile = bool(self.get_parameter('use_waypoint_speed_profile').value)
        self.config.speed_profile_scale = max(
            0.0,
            float(self.get_parameter('speed_profile_scale').value),
        )
        self.config.speed_profile_min_speed = max(
            0.0,
            float(self.get_parameter('speed_profile_min_speed').value),
        )
        self.config.speed_profile_max_speed = max(
            self.config.speed_profile_min_speed,
            float(self.get_parameter('speed_profile_max_speed').value),
        )
        self.config.speed_profile_lookahead_steps = max(
            0,
            int(self.get_parameter('speed_profile_lookahead_steps').value),
        )
        self.config.speed_profile_iterations = max(
            1,
            int(self.get_parameter('speed_profile_iterations').value),
        )
        self.config.use_speed_profile_drive_speed = bool(
            self.get_parameter('use_speed_profile_drive_speed').value
        )
        self.config.speed_profile_drive_blend = float(np.clip(
            float(self.get_parameter('speed_profile_drive_blend').value),
            0.0,
            1.0,
        ))
        self.config.speed_profile_drive_lookahead_steps = max(
            0,
            int(self.get_parameter('speed_profile_drive_lookahead_steps').value),
        )
        self.config.speed_profile_drive_use_min_lookahead = bool(
            self.get_parameter('speed_profile_drive_use_min_lookahead').value
        )
        self.config.speed_profile_drive_max_accel = max(
            0.0,
            float(self.get_parameter('speed_profile_drive_max_accel').value),
        )
        self.config.speed_profile_drive_max_decel = max(
            0.0,
            float(self.get_parameter('speed_profile_drive_max_decel').value),
        )
        self.config.min_speed = float(self.get_parameter('min_speed').value)
        self.config.max_speed = max(
            self.config.min_speed,
            float(self.get_parameter('max_speed').value),
        )
        self.config.max_steering_angle = float(self.get_parameter('max_steering_angle').value)
        self.config.publish_markers = bool(self.get_parameter('publish_markers').value)
        self.config.marker_frame_id = str(self.get_parameter('marker_frame_id').value)
        self.config.reference_line_width = max(
            0.001,
            float(self.get_parameter('reference_line_width').value),
        )
        self.config.optimal_line_width = max(
            0.001,
            float(self.get_parameter('optimal_line_width').value),
        )
        self.config.sampled_line_width = max(
            0.001,
            float(self.get_parameter('sampled_line_width').value),
        )
        self.config.sampled_trajectory_count = max(
            0,
            int(self.get_parameter('sampled_trajectory_count').value),
        )
        self.config.sampled_trajectory_alpha = float(np.clip(
            float(self.get_parameter('sampled_trajectory_alpha').value),
            0.0,
            1.0,
        ))

        if hasattr(self, 'infer_env'):
            self.infer_env.config = self.config
            self.infer_env.norm_params = self.config.norm_params
            self.infer_env._init_wall_cost()
        if hasattr(self, 'mppi'):
            self.mppi.temperature = self.config.temperature
            self.mppi.damping = self.config.damping
            self.mppi.n_iterations = self.config.n_iterations

        if hasattr(self, 'state_est_method_enabled'):
            if self.state_est_method_enabled != self.config.use_pose_delta_state_estimate:
                self.reset_state_estimator()
                self.state_est_method_enabled = bool(self.config.use_pose_delta_state_estimate)

    def resolve_map_dir(self):
        map_dir = Path(self.config.map_dir)
        if not map_dir.is_absolute():
            map_dir = (self.config_path.parent / map_dir).resolve()
        self.config.map_dir = str(map_dir) + os.sep

    @staticmethod
    def quaternion_to_yaw(quaternion):
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    @staticmethod
    def wrap_angle(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def stamp_to_sec(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def opponent_path_callback(self, msg):
        poses = list(msg.poses)
        if not poses:
            return

        start_idx = 1 if len(poses) > 1 else 0
        points = []
        for stamped_pose in poses[start_idx:start_idx + self.config.n_steps]:
            point = stamped_pose.pose.position
            if np.isfinite(point.x) and np.isfinite(point.y):
                points.append([float(point.x), float(point.y)])

        if not points:
            return

        horizon = np.asarray(points, dtype=np.float32)
        if horizon.shape[0] < self.config.n_steps:
            last = horizon[-1:, :]
            pad = np.repeat(last, self.config.n_steps - horizon.shape[0], axis=0)
            horizon = np.vstack([horizon, pad])

        self.opponent_xy_horizon = horizon[:self.config.n_steps]
        self.opponent_path_time = time.time()

    def get_opponent_horizon(self):
        horizon = np.asarray(self.opponent_xy_horizon, dtype=np.float32)
        if horizon.shape != (self.config.n_steps, 2):
            fixed = np.zeros((self.config.n_steps, 2), dtype=np.float32)
            rows = min(horizon.shape[0], self.config.n_steps) if horizon.ndim == 2 else 0
            if rows > 0:
                fixed[:rows, :] = horizon[:rows, :2]
                fixed[rows:, :] = fixed[rows - 1:rows, :]
            horizon = fixed

        if self.opponent_path_time is None:
            return horizon, False, np.inf

        age = time.time() - self.opponent_path_time
        active = (
            self.config.opponent_cost_enabled
            and age <= self.config.opponent_path_timeout
            and np.isfinite(horizon).all()
        )
        return horizon, active, age

    def reset_state_estimator(self):
        self.prev_pose_time = None
        self.prev_pose_xy = None
        self.prev_pose_yaw = None
        self.state_est_vx = None
        self.state_est_vy = 0.0
        self.state_est_wz = 0.0

    def estimate_vehicle_state(self, pose, theta, twist, callback_time):
        raw_vx = float(twist.linear.x)
        raw_vy = float(twist.linear.y)
        raw_wz = float(twist.angular.z)
        raw_beta = np.arctan2(raw_vy, max(abs(raw_vx), 1e-6))

        if self.config.is_sim or not self.config.use_pose_delta_state_estimate:
            return max(raw_vx, self.config.init_vel), raw_vy, raw_wz, raw_beta

        pose_xy = np.asarray([pose.position.x, pose.position.y], dtype=float)
        commanded_speed = float(np.clip(
            self.control[1],
            self.config.min_speed,
            self.config.max_speed,
        ))

        if (
            self.prev_pose_time is None
            or self.prev_pose_xy is None
            or self.prev_pose_yaw is None
        ):
            self.prev_pose_time = callback_time
            self.prev_pose_xy = pose_xy
            self.prev_pose_yaw = theta
            seed_speed = max(raw_vx, commanded_speed, self.config.init_vel)
            self.state_est_vx = seed_speed
            self.state_est_vy = raw_vy
            self.state_est_wz = raw_wz
            return seed_speed, raw_vy, raw_wz, np.arctan2(raw_vy, max(seed_speed, 1e-6))

        dt = callback_time - self.prev_pose_time
        if not np.isfinite(dt) or dt <= 1e-3 or dt > 0.5:
            self.prev_pose_time = callback_time
            self.prev_pose_xy = pose_xy
            self.prev_pose_yaw = theta
            fallback_vx = max(
                raw_vx,
                commanded_speed,
                self.state_est_vx if self.state_est_vx is not None else self.config.init_vel,
            )
            fallback_vy = self.state_est_vy
            fallback_wz = self.state_est_wz
            return fallback_vx, fallback_vy, fallback_wz, np.arctan2(
                fallback_vy,
                max(abs(fallback_vx), 1e-6),
            )

        world_vel = (pose_xy - self.prev_pose_xy) / dt
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        vx_pose = cos_th * world_vel[0] + sin_th * world_vel[1]
        vy_pose = -sin_th * world_vel[0] + cos_th * world_vel[1]
        wz_pose = self.wrap_angle(theta - self.prev_pose_yaw) / dt

        vx_pose = float(np.clip(vx_pose, -self.config.max_speed, self.config.max_speed))
        vy_pose = float(np.clip(vy_pose, -3.0, 3.0))
        wz_pose = float(np.clip(wz_pose, -8.0, 8.0))

        speed_obs_terms = [vx_pose]
        if np.isfinite(raw_vx):
            speed_obs_terms.append(raw_vx)
        speed_obs = float(np.mean(speed_obs_terms))
        prev_vx = self.state_est_vx if self.state_est_vx is not None else speed_obs
        vx_est = 0.55 * speed_obs + 0.30 * commanded_speed + 0.15 * prev_vx
        vx_est = float(np.clip(vx_est, self.config.min_speed, self.config.max_speed))
        vx_est = max(vx_est, self.config.init_vel)

        vy_obs = 0.8 * vy_pose + 0.2 * raw_vy
        vy_est = 0.35 * vy_obs + 0.65 * self.state_est_vy
        vy_est = float(np.clip(vy_est, -2.0, 2.0))

        wz_obs = 0.85 * wz_pose + 0.15 * raw_wz
        wz_est = 0.45 * wz_obs + 0.55 * self.state_est_wz
        wz_est = float(np.clip(wz_est, -6.0, 6.0))

        self.prev_pose_time = callback_time
        self.prev_pose_xy = pose_xy
        self.prev_pose_yaw = theta
        self.state_est_vx = vx_est
        self.state_est_vy = vy_est
        self.state_est_wz = wz_est

        beta_est = np.arctan2(vy_est, max(abs(vx_est), 1e-6))
        return vx_est, vy_est, wz_est, beta_est

    @staticmethod
    def make_point(x, y, z=0.05):
        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = float(z)
        return point

    def make_line_strip_marker(self, ns, marker_id, xy_points, rgba, width, stamp, z=0.05):
        xy_points = np.asarray(xy_points)
        if xy_points.ndim == 1:
            xy_points = xy_points.reshape(1, -1)

        marker = Marker()
        marker.header.frame_id = self.config.marker_frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = int(marker_id)
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(width)
        marker.color.r = float(rgba[0])
        marker.color.g = float(rgba[1])
        marker.color.b = float(rgba[2])
        marker.color.a = float(rgba[3])
        marker.points = [
            self.make_point(point[0], point[1], z)
            for point in xy_points
            if point.shape[0] >= 2 and np.isfinite(point[0]) and np.isfinite(point[1])
        ]
        return marker

    def make_delete_all_marker(self, ns, stamp):
        marker = Marker()
        marker.header.frame_id = self.config.marker_frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.action = Marker.DELETEALL
        return marker

    def mppi_state_trajectory_to_xy(self, trajectory):
        trajectory = np.asarray(trajectory)
        if trajectory.size == 0:
            return np.empty((0, 2))

        if self.mppi.track is None or trajectory.shape[-1] < 5:
            return trajectory[..., :2]

        flat = trajectory.reshape(-1, trajectory.shape[-1])
        xy = np.empty((flat.shape[0], 2), dtype=float)
        for i, state in enumerate(flat):
            x, y, _ = self.mppi.track.frenet_to_cartesian(state[0], state[1], state[4])
            xy[i] = [x, y]
        return xy.reshape(*trajectory.shape[:-1], 2)

    def publish_visualization(self, reference_traj):
        if not self.config.publish_markers:
            return

        stamp = self.get_clock().now().to_msg()

        if self.reference_marker_pub.get_subscription_count() > 0:
            reference_xy = np.asarray(reference_traj)[..., :2]
            marker_array = MarkerArray()
            marker_array.markers.append(self.make_line_strip_marker(
                'mppi_reference',
                0,
                reference_xy,
                (0.05, 0.35, 1.0, 1.0),
                self.config.reference_line_width,
                stamp,
                z=0.06,
            ))
            self.reference_marker_pub.publish(marker_array)

        if self.opt_traj_marker_pub.get_subscription_count() > 0:
            opt_traj_xy = self.mppi_state_trajectory_to_xy(numpify(self.mppi.traj_opt))
            marker_array = MarkerArray()
            marker_array.markers.append(self.make_line_strip_marker(
                'mppi_optimal_trajectory',
                0,
                opt_traj_xy,
                (0.0, 0.95, 0.2, 1.0),
                self.config.optimal_line_width,
                stamp,
                z=0.08,
            ))
            self.opt_traj_marker_pub.publish(marker_array)

        if self.sampled_traj_marker_pub.get_subscription_count() > 0:
            marker_array = MarkerArray()
            marker_array.markers.append(self.make_delete_all_marker('mppi_sampled_trajectories', stamp))

            max_count = self.config.sampled_trajectory_count
            if max_count > 0:
                sampled_xy = self.mppi_state_trajectory_to_xy(numpify(self.mppi.sampled_states))
                if sampled_xy.ndim == 3 and sampled_xy.shape[0] > 0:
                    stride = max(1, int(np.ceil(sampled_xy.shape[0] / max_count)))
                    for marker_id, trajectory_xy in enumerate(sampled_xy[::stride][:max_count]):
                        marker_array.markers.append(self.make_line_strip_marker(
                            'mppi_sampled_trajectories',
                            marker_id,
                            trajectory_xy,
                            (1.0, 0.42, 0.05, self.config.sampled_trajectory_alpha),
                            self.config.sampled_line_width,
                            stamp,
                            z=0.04,
                        ))
            self.sampled_traj_marker_pub.publish(marker_array)

    def publish_reward_debug(self, reference_traj, opponent_traj=None, opponent_active=False, opponent_age=np.inf):
        if not any(pub.get_subscription_count() > 0 for pub in self.reward_debug_pubs.values()):
            return

        reward_weights = np.asarray([
            self.config.xy_reward_weight,
            self.config.velocity_reward_weight,
            self.config.yaw_reward_weight,
        ], dtype=np.float32)
        cost_params = np.asarray([
            self.config.wall_cost_weight if self.config.wall_cost_enabled else 0.0,
            self.config.wall_cost_margin,
            self.config.wall_cost_power,
            self.config.slip_cost_weight if self.config.slip_cost_enabled else 0.0,
            self.config.slip_cost_beta_safe,
            self.config.latacc_cost_weight if self.config.latacc_cost_enabled else 0.0,
            self.config.latacc_cost_safe,
            self.config.steer_sat_cost_weight if self.config.steer_sat_cost_enabled else 0.0,
            self.config.steer_sat_soft_ratio * self.config.max_steering_angle,
            self.config.opponent_cost_weight if (self.config.opponent_cost_enabled and opponent_active) else 0.0,
            self.config.opponent_cost_radius,
            self.config.opponent_cost_power,
            self.config.opponent_cost_discount,
        ], dtype=np.float32)

        debug_terms = self.infer_env.reward_debug_terms(
            numpify(self.mppi.traj_opt),
            numpify(reference_traj),
            reward_weights=reward_weights,
            cost_params=cost_params,
            opponent_traj=opponent_traj,
        )
        debug_terms['opponent_path_age'] = float(opponent_age) if np.isfinite(opponent_age) else -1.0
        debug_terms['opponent_active'] = 1.0 if opponent_active else 0.0
        if not opponent_active:
            debug_terms['min_opponent_dist'] = -1.0

        for key, pub in self.reward_debug_pubs.items():
            if pub.get_subscription_count() == 0:
                continue
            msg = Float32()
            msg.data = float(debug_terms.get(key, 0.0))
            pub.publish(msg)

    def pose_callback(self, pose_msg):
        """
        Callback function for subscribing to particle filter's  inferred pose.
        This funcion saves the current pose of the car and obtain the goal
        waypoint from the pure pursuit module.

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        """
        t1 = time.time()
        self.get_params()
        pose = pose_msg.pose.pose
        twist = pose_msg.twist.twist
        theta = self.quaternion_to_yaw(pose.orientation)
        callback_time = self.stamp_to_sec(pose_msg.header.stamp)
        if callback_time <= 0.0:
            callback_time = t1
        vx_state, vy_state, wz_state, beta = self.estimate_vehicle_state(
            pose,
            theta,
            twist,
            callback_time,
        )

        state_c_0 = np.asarray([
            pose.position.x,
            pose.position.y,
            self.control[0],
            vx_state,
            theta,
            wz_state,
            beta,
        ])
        find_waypoint_vel = max(self.config.ref_vel, vx_state)
        
        reference_traj, waypoint_ind = self.infer_env.get_refernece_traj(state_c_0, find_waypoint_vel, self.config.n_steps)
        opponent_traj, opponent_active, opponent_age = self.get_opponent_horizon()

        ## MPPI call
        self.mppi.update(
            jnp.asarray(state_c_0),
            jnp.asarray(reference_traj),
            jnp.asarray(opponent_traj),
            opponent_active,
        )
        mppi_control = numpify(self.mppi.a_opt[0]) * self.config.norm_params[0, :2]/2
        prev_speed_command = float(self.control[1])
        mppi_speed_command = float(mppi_control[1]) * self.config.sim_time_step + vx_state
        profile_speed_command = np.nan
        speed_command = mppi_speed_command

        if self.config.use_waypoint_speed_profile and self.config.use_speed_profile_drive_speed:
            speed_idx = min(
                self.config.speed_profile_drive_lookahead_steps,
                reference_traj.shape[0] - 1,
            )
            if self.config.speed_profile_drive_use_min_lookahead:
                profile_speed_command = float(np.min(reference_traj[:speed_idx + 1, 2]))
            else:
                profile_speed_command = float(reference_traj[speed_idx, 2])
            blend = self.config.speed_profile_drive_blend
            if np.isnan(mppi_speed_command):
                mppi_speed_command = prev_speed_command
            if blend >= 1.0 - 1e-6:
                speed_command = profile_speed_command
            else:
                speed_command = (1.0 - blend) * mppi_speed_command + blend * profile_speed_command

            if self.last_speed_command_time is None:
                speed_command_dt = self.config.sim_time_step
            else:
                speed_command_dt = max(0.0, t1 - self.last_speed_command_time)
            max_accel_step = self.config.speed_profile_drive_max_accel * speed_command_dt
            max_decel_step = self.config.speed_profile_drive_max_decel * speed_command_dt
            if max_accel_step > 0.0 and speed_command > prev_speed_command:
                speed_command = min(speed_command, prev_speed_command + max_accel_step)
            if max_decel_step > 0.0 and speed_command < prev_speed_command:
                speed_command = max(speed_command, prev_speed_command - max_decel_step)
        self.last_speed_command_time = t1

        self.control[0] = float(mppi_control[0]) * self.config.sim_time_step + self.control[0]
        self.control[1] = speed_command
        self.control[0] = float(np.clip(
            self.control[0],
            -self.config.max_steering_angle,
            self.config.max_steering_angle,
        ))
        self.control[1] = float(np.clip(
            self.control[1],
            self.config.min_speed,
            self.config.max_speed,
        ))
        
        if self.reference_pub.get_subscription_count() > 0:
            ref_traj_cpu = numpify(reference_traj)
            arr_msg = to_multiarray_f32(ref_traj_cpu.astype(np.float32))
            self.reference_pub.publish(arr_msg)

        if self.opt_traj_pub.get_subscription_count() > 0:
            opt_traj_cpu = numpify(self.mppi.traj_opt)
            arr_msg = to_multiarray_f32(opt_traj_cpu.astype(np.float32))
            self.opt_traj_pub.publish(arr_msg)

        self.publish_reward_debug(reference_traj, opponent_traj, opponent_active, opponent_age)
        self.publish_visualization(reference_traj)

        if vx_state < self.config.init_vel:
            startup_speed = np.clip(
                self.config.startup_speed,
                self.config.min_speed,
                self.config.max_speed,
            )
            self.control[1] = float(max(self.control[1], startup_speed))

        if self.speed_debug_pub.get_subscription_count() > 0:
            speed_debug = np.array([
                vx_state,
                mppi_speed_command,
                profile_speed_command,
                self.control[1],
                self.config.speed_profile_drive_blend,
            ], dtype=np.float32)
            self.speed_debug_pub.publish(to_multiarray_f32(speed_debug))

        if np.isnan(self.control).any() or np.isinf(self.control).any():
            self.control = np.array([0.0, 0.0])
            self.mppi.a_opt = np.zeros_like(self.mppi.a_opt)

        # Publish the control command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = self.control[0]
        drive_msg.drive.speed = self.control[1]
        # self.get_logger().info(f"Steering Angle: {drive_msg.drive.steering_angle}, Speed: {drive_msg.drive.speed}")
        self.drive_pub.publish(drive_msg)
        self.hz.append(1/(time.time() - t1))
        if len(self.hz) == 100:
            self.hz = np.mean(self.hz)
            print(f"MPPI Hz: {self.hz:.2f}")
            self.hz = []
        

def main(args=None):
    rclpy.init(args=args)
    mppi_node = MPPI_Node()
    rclpy.spin(mppi_node)

    mppi_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
