import os
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
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
#jax.config.update("jax_compilation_cache_dir", "/home/nvidia/jax_cache") 


## This is a demosntration of how to use the MPPI planner with the Roboracer
## Zirui Zang 2025/04/07

class MPPI_Node(Node):
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
        
        self.mppi.update(jnp.asarray(state_c_0), jnp.asarray(reference_traj))
        self.get_logger().info('MPPI initialized')
        self.hz = []
        self.last_speed_command_time = None
        
        
        qos = rclpy.qos.QoSProfile(history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
                                   depth=1,
                                   reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
                                   durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE)
        # create subscribers
        if self.config.is_sim:
            self.pose_sub = self.create_subscription(Odometry, "/ego_racecar/odom", self.pose_callback, qos)
        else:
            self.pose_sub = self.create_subscription(Odometry, "/pf/pose/odom", self.pose_callback, qos)
        # publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", qos)
        self.reference_pub = self.create_publisher(Float32MultiArray, "/reference_arr", qos)
        self.opt_traj_pub = self.create_publisher(Float32MultiArray, "/opt_traj_arr", qos)
        self.reference_marker_pub = self.create_publisher(MarkerArray, "/mppi/reference", qos)
        self.opt_traj_marker_pub = self.create_publisher(MarkerArray, "/mppi/optimal_trajectory", qos)
        self.sampled_traj_marker_pub = self.create_publisher(MarkerArray, "/mppi/sampled_trajectories", qos)
        self.speed_debug_pub = self.create_publisher(Float32MultiArray, "/mppi/speed_debug", qos)

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
            'friction': float(self.config.friction),
            'n_iterations': int(self.config.n_iterations),
            'control_sample_std_steer': float(self.config.control_sample_std[0]),
            'control_sample_std_accel': float(self.config.control_sample_std[1]),
            'steer_vel_scale': steer_vel_scale,
            'accel_scale': accel_scale,
            'xy_reward_weight': float(self.config.xy_reward_weight),
            'velocity_reward_weight': float(self.config.velocity_reward_weight),
            'yaw_reward_weight': float(self.config.yaw_reward_weight),
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
        if hasattr(self, 'mppi'):
            self.mppi.temperature = self.config.temperature
            self.mppi.damping = self.config.damping
            self.mppi.n_iterations = self.config.n_iterations

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

        # Beta calculated by the arctan of the lateral velocity and the longitudinal velocity
        beta = np.arctan2(twist.linear.y, twist.linear.x)

        theta = self.quaternion_to_yaw(pose.orientation)

        state_c_0 = np.asarray([
            pose.position.x,
            pose.position.y,
            self.control[0],
            max(twist.linear.x, self.config.init_vel),
            theta,
            twist.angular.z,
            beta,
        ])
        find_waypoint_vel = max(self.config.ref_vel, twist.linear.x)
        
        reference_traj, waypoint_ind = self.infer_env.get_refernece_traj(state_c_0, find_waypoint_vel, self.config.n_steps)

        ## MPPI call
        self.mppi.update(jnp.asarray(state_c_0), jnp.asarray(reference_traj))
        mppi_control = numpify(self.mppi.a_opt[0]) * self.config.norm_params[0, :2]/2
        prev_speed_command = float(self.control[1])
        mppi_speed_command = float(mppi_control[1]) * self.config.sim_time_step + twist.linear.x
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

        self.publish_visualization(reference_traj)

        if twist.linear.x < self.config.init_vel:
            startup_speed = np.clip(
                self.config.startup_speed,
                self.config.min_speed,
                self.config.max_speed,
            )
            self.control[1] = float(max(self.control[1], startup_speed))

        if self.speed_debug_pub.get_subscription_count() > 0:
            speed_debug = np.array([
                twist.linear.x,
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
