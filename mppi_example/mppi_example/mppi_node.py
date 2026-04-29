import os
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from nav_msgs.msg import Odometry, Path as NavPath
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from std_msgs.msg import Float32, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from raceline_msgs.srv import UpdateRaceline
from rcl_interfaces.msg import ParameterDescriptor, FloatingPointRange, IntegerRange

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
        'cost_opponent_follow_sum': '/mppi/debug/cost_opponent_follow_sum',
        'cost_opponent_pass_sum': '/mppi/debug/cost_opponent_pass_sum',
        'min_wall_dist': '/mppi/debug/min_wall_dist',
        'min_opponent_dist': '/mppi/debug/min_opponent_dist',
        'min_opponent_longitudinal_rel': '/mppi/debug/min_opponent_longitudinal_rel',
        'max_abs_opponent_lateral_rel': '/mppi/debug/max_abs_opponent_lateral_rel',
        'opponent_auto_mode_id': '/mppi/debug/opponent_auto_mode_id',
        'opponent_auto_left_clearance': '/mppi/debug/opponent_auto_left_clearance',
        'opponent_auto_right_clearance': '/mppi/debug/opponent_auto_right_clearance',
        'opponent_auto_closing_speed': '/mppi/debug/opponent_auto_closing_speed',
        'opponent_auto_pass_allowed': '/mppi/debug/opponent_auto_pass_allowed',
        'opponent_path_age': '/mppi/debug/opponent_path_age',
        'opponent_active': '/mppi/debug/opponent_active',
        'callback_wall_dt': '/mppi/debug/callback_wall_dt',
        'callback_stamp_dt': '/mppi/debug/callback_stamp_dt',
        'prev_pose_dt': '/mppi/debug/prev_pose_dt',
        'state_est_vy': '/mppi/debug/state_est_vy',
        'state_est_wz': '/mppi/debug/state_est_wz',
        'mppi_solve_time': '/mppi/debug/mppi_solve_time',
        'mppi_aopt_max_abs': '/mppi/debug/mppi_aopt_max_abs',
        'mppi_saturation_count': '/mppi/debug/mppi_saturation_count',
        'mppi_bad_output_count': '/mppi/debug/mppi_bad_output_count',
        'mppi_guard_count': '/mppi/debug/mppi_guard_count',
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
        self.last_speed_command_time = None
        self.last_pose_callback_wall_time = None
        self.last_pose_msg_time = None
        self.mppi_guard_count = 0          # HARD guards (warm-start wiped)
        self.mppi_soft_guard_count = 0     # SOFT guards (timing gap noted, warm-start kept)
        self.mppi_saturation_count = 0
        self.mppi_bad_output_count = 0

        # Timer-driven control loop state. pose_callback only writes
        # latest_pose_msg + recv_time; control_timer reads them. CPython
        # attribute assignment is atomic, so a single-pointer swap is race-free.
        self.latest_pose_msg = None
        self.latest_pose_recv_time = None
        self.last_control_step_wall_time = None
        # Visualization-only rollout. Populated when use_pose_delta_state_estimate
        # is true so the marker shows a clean (raw-twist-rolled) trajectory
        # while the controller still solved against the IIR-estimated state.
        # publish_visualization prefers this when non-None.
        self.viz_traj_opt = None
        # Rolling stats for the periodic logger.
        self._stats_window_start = time.time()
        self._stats_pose_rx = 0
        self._stats_drive_tx = 0
        self._stats_control_ticks = 0
        self._stats_control_skips_stale = 0
        self._stats_control_skips_no_pose = 0
        self._stats_solve_times = []
        self._stats_pose_age = 0.0
        self._stats_guard_count_at_window_start = 0
        self._stats_soft_guard_count_at_window_start = 0
        self._stats_opponent_rx = 0
        self.last_timing_debug = {
            'callback_wall_dt': 0.0,
            'callback_stamp_dt': 0.0,
            'prev_pose_dt': 0.0,
            'state_est_vy': 0.0,
            'state_est_wz': 0.0,
            'mppi_solve_time': 0.0,
            'mppi_aopt_max_abs': 0.0,
        }
        self.prev_pose_time = None
        self.prev_pose_xy = None
        self.prev_pose_yaw = None
        self.state_est_vx = None
        self.state_est_vy = 0.0
        self.state_est_wz = 0.0
        self.state_est_method_enabled = bool(self.config.use_pose_delta_state_estimate)
        self.opponent_auto_debug = {
            'mode_id': 0.0,
            'left_clearance': -1.0,
            'right_clearance': -1.0,
            'closing_speed': 0.0,
            'pass_allowed': 0.0,
        }
        # Auto-overtake state machine. Stateless re-evaluation every callback
        # caused mode flicker at boundary conditions (closing speed dipping
        # near min, clearance crossing threshold mid-pass). Track which side
        # we committed to and when, so a brief dip doesn't abort the pass.
        self.auto_state = 'idle'           # idle | follow | pass_left | pass_right
        self.auto_state_entered_at = None  # wall time the current state was entered
        self._auto_warned_no_wall_sdf = False

        
        qos = rclpy.qos.QoSProfile(
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
        )
        sensor_qos = rclpy.qos.QoSProfile(
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
        )
        debug_qos = rclpy.qos.QoSProfile(
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
        )
        # Reentrant group so pose_callback (cache-only) and control_timer
        # (the solve) can run concurrently under the MultiThreadedExecutor.
        # opponent_path_callback joins the same group; it touches independent
        # state. Service + stats_timer stay in the default group.
        self._control_group = ReentrantCallbackGroup()
        # create subscribers
        if self.config.is_sim:
            self.pose_sub = self.create_subscription(
                Odometry, "/ego_racecar/odom", self.pose_callback,
                sensor_qos, callback_group=self._control_group,
            )
        else:
            self.pose_sub = self.create_subscription(
                Odometry, "/pf/pose/odom", self.pose_callback,
                sensor_qos, callback_group=self._control_group,
            )
        self.opponent_path_sub = self.create_subscription(
            NavPath,
            self.config.opponent_path_topic,
            self.opponent_path_callback,
            sensor_qos,
            callback_group=self._control_group,
        )
        # publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", qos)
        self.reference_pub = self.create_publisher(Float32MultiArray, "/reference_arr", debug_qos)
        self.opt_traj_pub = self.create_publisher(Float32MultiArray, "/opt_traj_arr", debug_qos)
        self.reference_marker_pub = self.create_publisher(MarkerArray, "/mppi/reference", debug_qos)
        self.opt_traj_marker_pub = self.create_publisher(MarkerArray, "/mppi/optimal_trajectory", debug_qos)
        self.sampled_traj_marker_pub = self.create_publisher(MarkerArray, "/mppi/sampled_trajectories", debug_qos)
        self.speed_debug_pub = self.create_publisher(Float32MultiArray, "/mppi/speed_debug", debug_qos)
        self.reward_debug_pubs = {
            key: self.create_publisher(Float32, topic, debug_qos)
            for key, topic in self.DEBUG_SCALAR_TOPICS.items()
        }

        self.update_raceline_srv = self.create_service(
            UpdateRaceline, '/mppi/update_raceline', self.update_raceline_callback
        )
        self.get_logger().info('Raceline update service ready at /mppi/update_raceline')

        # Fixed-rate control loop. Period derived from control_loop_hz at
        # startup (read_only param so we do not need to re-create the timer).
        control_period = 1.0 / max(1.0, float(self.config.control_loop_hz))
        self.control_loop_timer = self.create_timer(
            control_period, self.control_timer, callback_group=self._control_group,
        )
        self.get_logger().info(
            f'Control loop timer at {self.config.control_loop_hz:.1f} Hz '
            f'(period {control_period*1000:.1f} ms)'
        )

        # Periodic stats logger. Default group so it never competes with the
        # control loop for callback-group slots.
        if self.config.stats_log_interval_sec > 0.0:
            self.stats_log_timer = self.create_timer(
                self.config.stats_log_interval_sec, self.stats_timer,
            )

        # Refresh live params at 2 Hz instead of inside control_step. The
        # rclpy parameter service is mutex-protected pure-Python; reading
        # ~50 params at 40 Hz held the GIL ~20%+ of wall time, head-of-line
        # blocking pose_callback under the MultiThreadedExecutor. Default
        # callback group keeps it off the control hot path.
        self.params_refresh_timer = self.create_timer(0.5, self.get_params)

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
            'sampled_trajectory_count': 0,
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
            'opponent_behavior_mode': 'follow',
            'opponent_behavior_mode_id': 0.0,
            'opponent_follow_weight': 0.0,
            'opponent_follow_distance': 1.2,
            'opponent_same_lane_width': 0.7,
            'opponent_pass_weight': 0.0,
            'opponent_pass_lateral_offset': 0.55,
            'opponent_pass_longitudinal_window': 1.5,
            'opponent_auto_wall_check_enabled': True,
            'opponent_auto_min_wall_clearance': 0.45,    # ENTER clearance (decide to pass)
            'opponent_auto_check_steps': 3,
            'opponent_auto_min_closing_speed': 0.4,      # ENTER closing speed
            'opponent_auto_max_ahead_distance': 4.0,
            'opponent_auto_side_switch_margin': 0.08,
            # Hysteresis: looser exit thresholds so a transient dip mid-pass
            # doesn't abort a committed overtake. Defaults are 0 -> derived
            # from the enter values when not explicitly set (see get_params).
            'opponent_auto_exit_wall_clearance': 0.0,
            'opponent_auto_exit_closing_speed': 0.0,
            'opponent_auto_min_commit_sec': 0.5,         # min time to stay committed once a side is chosen
            'slip_cost_enabled': False,
            'slip_cost_weight': 0.0,
            'slip_cost_beta_safe': 0.2,
            'latacc_cost_enabled': False,
            'latacc_cost_weight': 0.0,
            'latacc_cost_safe': 8.0,
            'steer_sat_cost_enabled': False,
            'steer_sat_cost_weight': 0.0,
            'steer_sat_soft_ratio': 0.85,
            'mppi_guard_on_timing_jump': True,
            'mppi_guard_wall_gap': 0.25,
            'mppi_guard_stamp_gap': 0.25,
            # Hard reset threshold: only WIPE the warm-start when a real
            # stall happens (process freeze, JAX recompile, multi-second
            # disk I/O spike). For moderate PF jitter (above the soft
            # wall_gap/stamp_gap thresholds but below this), continuity is
            # more valuable than starting fresh, so we only log/count.
            'mppi_guard_hard_gap': 1.5,
            'mppi_guard_aopt_threshold': 0.98,
            'mppi_guard_saturation_callbacks': 4,
            'mppi_guard_bad_callbacks_to_clear_control': 3,
            'state_est_vy_prior': 0.40,
            'state_est_wz_prior': 0.40,
            'state_est_hiccup_dt': 0.06,
            'state_est_hiccup_prior_scale': 0.35,
            # Timer-driven control loop. MPPI ticks at fixed rate independent of
            # PF cadence. Required so /scan or PF degradation under load
            # (rosbag, opponent node) does not slow the controller down with it.
            'control_loop_hz': 25.0,
            'control_pose_stale_sec': 0.20,
            'stats_log_interval_sec': 5.0,

        }
        for key, value in defaults.items():
            if not hasattr(self.config, key):
                setattr(self.config, key, value)

    def declare_ros_params(self):
        def fdesc(desc, lo, hi, step=0.0, read_only=False):
            # NOTE on step: rqt uses step to decide how many decimals to show
            # (step=0.001 -> 3 decimals, step=0.01 -> 2, step=1.0 -> 0). ROS2
            # validates that (value - lo) is an integer multiple of step
            # (math.isclose, rel_tol=1e-9), so pick lo + step so YAML defaults
            # land on the grid. Use step=0.0 only when alignment isn't possible.
            return ParameterDescriptor(
                description=desc,
                floating_point_range=[FloatingPointRange(
                    from_value=float(lo), to_value=float(hi), step=float(step))],
                read_only=read_only,
            )

        def idesc(desc, lo, hi, step=1, read_only=False):
            return ParameterDescriptor(
                description=desc,
                integer_range=[IntegerRange(
                    from_value=int(lo), to_value=int(hi), step=int(step))],
                read_only=read_only,
            )

        def desc(text, read_only=False):
            return ParameterDescriptor(description=text, read_only=read_only)

        def declf(name, value, d):
            self.declare_parameter(name, float(value), d)

        def decli(name, value, d):
            self.declare_parameter(name, int(value), d)

        def declb(name, value, d):
            self.declare_parameter(name, bool(value), d)

        def decls(name, value, d):
            self.declare_parameter(name, str(value), d)

        steer_vel_scale = float(self.config.norm_params[0, 0] / 2.0)
        accel_scale = float(self.config.norm_params[0, 1] / 2.0)
        random_seed = -1 if self.config.random_seed is None else int(self.config.random_seed)

        # Order below intentionally mirrors params_realev_overtake.yaml so rqt
        # displays params in the same order as the YAML reads (rclpy preserves
        # insertion order, and rqt iterates that for layout).

        # ---- Startup (read once; restart to change) ----
        declb('is_sim', self.config.is_sim,
              desc('[startup] true: /ego_racecar/odom; false: /pf/pose/odom.', read_only=True))
        declb('wpt_path_absolute', self.config.wpt_path_absolute,
              desc('[startup] Use wpt_path as absolute path to raceline CSV.', read_only=True))
        decls('wpt_path', self.config.wpt_path,
              desc('[startup] Absolute path to MPPI raceline CSV.', read_only=True))
        decls('map_dir', self.config.map_dir,
              desc('[startup] Directory containing map_info.txt and raceline csv.', read_only=True))
        decli('map_ind', self.config.map_ind,
              idesc('[startup] Row index in map_info.txt to use.', 0, 64, read_only=True))
        decls('state_predictor', self.config.state_predictor,
              desc('[startup] Rollout model: dynamic_ST or kinematic_ST.', read_only=True))
        decli('n_samples', self.config.n_samples,
              idesc('[startup] Number of sampled control sequences (JAX recompile).',
                    16, 8192, read_only=True))
        decli('n_steps', self.config.n_steps,
              idesc('[startup] Rollout horizon length (JAX recompile).',
                    2, 64, read_only=True))
        declf('sim_time_step', self.config.sim_time_step,
              fdesc('[startup] Rollout integration timestep (s).',
                    0.01, 0.5, 0.005, read_only=True))
        decli('random_seed', random_seed,
              idesc('[startup] Sampling seed (-1 = random each run).',
                    -1, 1_000_000, read_only=True))
        declb('render', self.config.render,
              desc('[startup] Keep optimal/sampled rollouts available for visualization.',
                   read_only=True))

        # ---- Live tuning ----
        declf('temperature', self.config.temperature,
              fdesc('MPPI greediness. Lower=winner-take-all; higher=smoother averaging.',
                    0.001, 1.0, 0.001))
        declf('damping', self.config.damping,
              fdesc('Weight normalization stabilizer when rewards are similar.',
                    0.0, 0.1, 1e-4))
        declf('ref_vel', self.config.ref_vel,
              fdesc('Legacy constant-speed reference seed (mostly bypassed when profile on).',
                    0.0, 20.0, 0.1))
        declf('init_vel', self.config.init_vel,
              fdesc('Min speed assumed by rollout model at startup/very low speed.',
                    0.0, 5.0, 0.1))
        declf('startup_speed', self.config.startup_speed,
              fdesc('Min /drive.speed while measured speed < init_vel.',
                    0.0, 5.0, 0.1))
        declb('use_pose_delta_state_estimate', self.config.use_pose_delta_state_estimate,
              desc('Hardware speed estimator from pose deltas (off in sim).'))
        declf('friction', self.config.friction,
              fdesc('Tire/friction belief for dynamic_ST rollouts.',
                    0.05, 1.5, 0.01))
        decli('n_iterations', self.config.n_iterations,
              idesc('MPPI update passes per odom callback.', 1, 10))

        # ---- Speed profile (rollout reference) ----
        declb('use_waypoint_speed_profile', self.config.use_waypoint_speed_profile,
              desc('Use raceline vx_mps as MPPI reference speed.'))
        declf('speed_profile_scale', self.config.speed_profile_scale,
              fdesc('Multiplies raceline vx_mps. Raise to push pace.',
                    0.0, 3.0, 0.01))
        declf('speed_profile_min_speed', self.config.speed_profile_min_speed,
              fdesc('Lower clamp for profile speed (m/s).', 0.0, 20.0, 0.1))
        declf('speed_profile_max_speed', self.config.speed_profile_max_speed,
              fdesc('Upper clamp for profile speed (m/s).', 0.0, 25.0, 0.1))
        decli('speed_profile_lookahead_steps', self.config.speed_profile_lookahead_steps,
              idesc('Planning brake lookahead (steps).', 0, 20))
        decli('speed_profile_iterations', self.config.speed_profile_iterations,
              idesc('Profile rebuild passes per call.', 1, 10))

        # ---- Speed profile (drive feedforward) ----
        declb('use_speed_profile_drive_speed', self.config.use_speed_profile_drive_speed,
              desc('Blend profile speed into final /drive.speed command.'))
        declf('speed_profile_drive_blend', self.config.speed_profile_drive_blend,
              fdesc('0 = pure MPPI accel; 1 = pure profile speed.', 0.0, 1.0, 0.01))
        decli('speed_profile_drive_lookahead_steps', self.config.speed_profile_drive_lookahead_steps,
              idesc('Command brake lookahead (future ref step).', 0, 20))
        declb('speed_profile_drive_use_min_lookahead',
              self.config.speed_profile_drive_use_min_lookahead,
              desc('Use min speed through lookahead window.'))
        declf('speed_profile_drive_max_accel', self.config.speed_profile_drive_max_accel,
              fdesc('Max commanded speed increase rate (m/s^2).', 0.0, 30.0, 0.1))
        declf('speed_profile_drive_max_decel', self.config.speed_profile_drive_max_decel,
              fdesc('Max commanded speed decrease rate (m/s^2).', 0.0, 30.0, 0.1))

        # ---- Control sampling ----
        declf('control_sample_std_steer', self.config.control_sample_std[0],
              fdesc('Std of normalized steering-rate noise. Higher = sharper exploration.',
                    0.0, 2.0, 0.01))
        declf('control_sample_std_accel', self.config.control_sample_std[1],
              fdesc('Std of normalized accel noise. Higher = harder accel/brake exploration.',
                    0.0, 5.0, 0.05))
        declf('steer_vel_scale', steer_vel_scale,
              fdesc('Converts normalized steering action to rad/s.',
                    0.1, 10.0, 0.05))
        declf('accel_scale', accel_scale,
              fdesc('Converts normalized accel action to m/s^2.',
                    0.1, 20.0, 0.1))

        # ---- Reward weights ----
        declf('xy_reward_weight', self.config.xy_reward_weight,
              fdesc('Path-tracking term: distance from rollout to reference XY.',
                    0.0, 5.0, 0.01))
        declf('velocity_reward_weight', self.config.velocity_reward_weight,
              fdesc('Penalizes mismatch to reference speed.', 0.0, 5.0, 0.01))
        declf('yaw_reward_weight', self.config.yaw_reward_weight,
              fdesc('Penalizes heading mismatch.', 0.0, 5.0, 0.01))

        # ---- Wall cost ----
        declb('wall_cost_enabled', self.config.wall_cost_enabled,
              desc('Enable wall SDF cost.'))
        declf('wall_cost_weight', self.config.wall_cost_weight,
              fdesc('Wall cost weight.', 0.0, 1000.0, 1.0))
        declf('wall_cost_margin', self.config.wall_cost_margin,
              fdesc('Distance below which wall cost activates (m).', 0.0, 1.5, 0.01))
        declf('wall_cost_power', self.config.wall_cost_power,
              fdesc('Wall cost exponent.', 1.0, 5.0, 0.1))
        decls('wall_cost_map_yaml', self.config.wall_cost_map_yaml,
              desc('[startup] Map YAML for wall SDF (injected by launch).', read_only=True))

        # ---- Opponent cost ----
        decls('opponent_path_topic', self.config.opponent_path_topic,
              desc('[startup] Topic publishing opponent predicted Path.', read_only=True))
        declb('opponent_cost_enabled', self.config.opponent_cost_enabled,
              desc('Enable opponent-aware MPPI cost.'))
        declf('opponent_cost_weight', self.config.opponent_cost_weight,
              fdesc('Opponent proximity cost weight.', 0.0, 1000.0, 1.0))
        declf('opponent_cost_radius', self.config.opponent_cost_radius,
              fdesc('Soft keep-out radius around opponent (m).', 0.0, 3.0, 0.01))
        declf('opponent_cost_power', self.config.opponent_cost_power,
              fdesc('Opponent cost exponent.', 1.0, 5.0, 0.1))
        declf('opponent_cost_discount', self.config.opponent_cost_discount,
              fdesc('Per-step discount along opponent horizon.', 0.0, 1.0, 0.01))
        declf('opponent_path_timeout', self.config.opponent_path_timeout,
              fdesc('Stale opponent path timeout (s).', 0.0, 5.0, 0.05))
        decls('opponent_behavior_mode', self.config.opponent_behavior_mode,
              desc('One of: follow | clear | pass_left | pass_right | auto.'))
        declf('opponent_follow_weight', self.config.opponent_follow_weight,
              fdesc('Follow mode: penalty for closing too near behind opponent.',
                    0.0, 1000.0, 0.5))
        declf('opponent_follow_distance', self.config.opponent_follow_distance,
              fdesc('Desired gap behind opponent (m).', 0.0, 5.0, 0.05))
        declf('opponent_same_lane_width', self.config.opponent_same_lane_width,
              fdesc('Follow gap only applies when ego roughly same lane (m).',
                    0.05, 3.0, 0.05))
        declf('opponent_pass_weight', self.config.opponent_pass_weight,
              fdesc('Pass mode: pushes rollout to requested side.', 0.0, 1000.0, 0.5))
        declf('opponent_pass_lateral_offset', self.config.opponent_pass_lateral_offset,
              fdesc('Desired lateral offset from opponent during pass (m).',
                    0.0, 2.0, 0.05))
        declb('opponent_auto_wall_check_enabled', self.config.opponent_auto_wall_check_enabled,
              desc('Auto mode: check wall clearance before passing.'))
        declf('opponent_auto_min_wall_clearance', self.config.opponent_auto_min_wall_clearance,
              fdesc('Min wall clearance to allow pass (m).', 0.0, 2.0, 0.01))
        decli('opponent_auto_check_steps', self.config.opponent_auto_check_steps,
              idesc('How many opponent horizon points to check for clearance.', 1, 20))
        declf('opponent_auto_min_closing_speed', self.config.opponent_auto_min_closing_speed,
              fdesc('Min ego-opponent closing speed to start a pass (m/s).',
                    0.0, 10.0, 0.1))
        declf('opponent_auto_max_ahead_distance', self.config.opponent_auto_max_ahead_distance,
              fdesc('Only consider passing opponents within this distance ahead (m).',
                    0.0, 20.0, 0.1))
        declf('opponent_auto_side_switch_margin', self.config.opponent_auto_side_switch_margin,
              fdesc('Hysteresis: extra clearance needed to flip preferred pass side (m).',
                    0.0, 1.0, 0.01))
        declf('opponent_auto_exit_wall_clearance', self.config.opponent_auto_exit_wall_clearance,
              fdesc('EXIT clearance — abort an in-progress pass if wall clearance falls below this. '
                    '0 means auto-derive from enter clearance (= 0.7x).', 0.0, 2.0, 0.01))
        declf('opponent_auto_exit_closing_speed', self.config.opponent_auto_exit_closing_speed,
              fdesc('EXIT closing speed — abort an in-progress pass if closing falls below this. '
                    '0 means auto-derive from enter closing (= 0.4x).', 0.0, 5.0, 0.05))
        declf('opponent_auto_min_commit_sec', self.config.opponent_auto_min_commit_sec,
              fdesc('Minimum time to stay committed once a pass side is chosen (s).',
                    0.0, 5.0, 0.05))
        declf('opponent_pass_longitudinal_window', self.config.opponent_pass_longitudinal_window,
              fdesc('Longitudinal window where pass-side cost applies (m).',
                    0.05, 5.0, 0.05))

        # ---- Slip cost ----
        declb('slip_cost_enabled', self.config.slip_cost_enabled,
              desc('Enable side-slip cost.'))
        declf('slip_cost_weight', self.config.slip_cost_weight,
              fdesc('Slip cost weight.', 0.0, 50.0, 0.1))
        declf('slip_cost_beta_safe', self.config.slip_cost_beta_safe,
              fdesc('Safe slip-angle threshold (rad).', 0.0, 1.5, 0.01))

        # ---- Lat-acc cost ----
        declb('latacc_cost_enabled', self.config.latacc_cost_enabled,
              desc('Enable lateral acceleration cost.'))
        declf('latacc_cost_weight', self.config.latacc_cost_weight,
              fdesc('Lat-acc cost weight.', 0.0, 50.0, 0.1))
        declf('latacc_cost_safe', self.config.latacc_cost_safe,
              fdesc('Lat-acc safe threshold (m/s^2).', 0.0, 100.0, 0.5))

        # ---- Steer-saturation cost ----
        declb('steer_sat_cost_enabled', self.config.steer_sat_cost_enabled,
              desc('Enable steering-saturation cost.'))
        declf('steer_sat_cost_weight', self.config.steer_sat_cost_weight,
              fdesc('Steering saturation cost weight.', 0.0, 10.0, 0.05))
        declf('steer_sat_soft_ratio', self.config.steer_sat_soft_ratio,
              fdesc('Soft limit as fraction of max steering.', 0.0, 1.0, 0.01))

        # ---- Output safety clamps ----
        declf('min_speed', self.config.min_speed,
              fdesc('Final /drive.speed lower clamp (m/s).', 0.0, 20.0, 0.1))
        declf('max_speed', self.config.max_speed,
              fdesc('Final /drive.speed upper clamp (m/s).', 0.0, 25.0, 0.1))
        declf('max_steering_angle', self.config.max_steering_angle,
              fdesc('Final steering-angle clamp (rad).', 0.0, 1.0, 0.0001))

        # ---- Visualization ----
        declb('publish_markers', self.config.publish_markers,
              desc('Toggle MPPI MarkerArray publishers.'))
        decls('marker_frame_id', self.config.marker_frame_id,
              desc('RViz frame for trajectory markers.'))
        declf('reference_line_width', self.config.reference_line_width,
              fdesc('Reference (blue) line width (m).', 0.0, 0.5, 0.005))
        declf('optimal_line_width', self.config.optimal_line_width,
              fdesc('Optimal rollout (green) line width (m).', 0.0, 0.5, 0.005))
        declf('sampled_line_width', self.config.sampled_line_width,
              fdesc('Sampled rollout (orange) line width (m).', 0.0, 0.5, 0.005))
        decli('sampled_trajectory_count', self.config.sampled_trajectory_count,
              idesc('Number of sampled rollouts to draw.', 0, 64))
        declf('sampled_trajectory_alpha', self.config.sampled_trajectory_alpha,
              fdesc('Sampled rollout transparency.', 0.0, 1.0, 0.01))
        declb('mppi_guard_on_timing_jump', self.config.mppi_guard_on_timing_jump,
              desc('Clear persistent MPPI/control state when odom/callback timing jumps.'))
        declf('mppi_guard_wall_gap', self.config.mppi_guard_wall_gap,
              fdesc('Wall-time callback gap that clears persistent MPPI/control state.', 0.05, 2.0, 0.01))
        declf('mppi_guard_stamp_gap', self.config.mppi_guard_stamp_gap,
              fdesc('SOFT odom stamp gap. Above this we count + log; below mppi_guard_hard_gap we KEEP the warm-start.', 0.05, 2.0, 0.01))
        declf('mppi_guard_hard_gap', self.config.mppi_guard_hard_gap,
              fdesc('HARD wall/stamp gap. Above this we wipe the warm-start (real stall).', 0.1, 5.0, 0.05))
        declf('mppi_guard_aopt_threshold', self.config.mppi_guard_aopt_threshold,
              fdesc('Clear warm-start if first-action components remain near saturation.', 0.5, 1.0, 0.01))
        decli('mppi_guard_saturation_callbacks', self.config.mppi_guard_saturation_callbacks,
              idesc('Consecutive saturated MPPI callbacks before clearing warm-start.', 1, 20))
        decli('mppi_guard_bad_callbacks_to_clear_control',
              self.config.mppi_guard_bad_callbacks_to_clear_control,
              idesc('Consecutive bad callbacks before falling back to startup speed.', 1, 10))
        declf('state_est_vy_prior', self.config.state_est_vy_prior,
              fdesc('Pose-delta vy IIR prior weight. Lower forgets bad samples faster.', 0.0, 0.95, 0.01))
        declf('state_est_wz_prior', self.config.state_est_wz_prior,
              fdesc('Pose-delta wz IIR prior weight. Lower forgets bad samples faster.', 0.0, 0.95, 0.01))
        declf('state_est_hiccup_dt', self.config.state_est_hiccup_dt,
              fdesc('Odom dt above this reduces vy/wz prior weights.', 0.0, 0.5, 0.005))
        declf('state_est_hiccup_prior_scale', self.config.state_est_hiccup_prior_scale,
              fdesc('Multiplier on vy/wz prior weights after an odom timing hiccup.', 0.0, 1.0, 0.01))
        # Timer-driven control loop. control_loop_hz is read at startup to
        # build the timer; pose-stale and stats-interval are live-tunable.
        declf('control_loop_hz', self.config.control_loop_hz,
              fdesc('[startup] Fixed-rate control loop frequency (Hz).', 1.0, 100.0, 1.0, read_only=True))
        declf('control_pose_stale_sec', self.config.control_pose_stale_sec,
              fdesc('Skip MPPI solve if cached pose age exceeds this (sec).', 0.05, 1.0, 0.01))
        declf('stats_log_interval_sec', self.config.stats_log_interval_sec,
              fdesc('Periodic stats logger interval (sec). 0 disables.', 0.0, 60.0, 0.5))

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
        mode_name = str(self.get_parameter('opponent_behavior_mode').value).strip().lower()
        mode_map = {
            'follow': 0.0,
            'auto': 0.0,
            'clear': 1.0,
            'pass_left': 2.0,
            'left': 2.0,
            'pass_right': 3.0,
            'right': 3.0,
        }
        if mode_name not in mode_map:
            self.get_logger().warn(
                f"Unknown opponent_behavior_mode='{mode_name}', using follow"
            )
            mode_name = 'follow'
        self.config.opponent_behavior_mode = mode_name
        self.config.opponent_behavior_mode_id = mode_map[mode_name]
        self.config.opponent_follow_weight = max(
            0.0,
            float(self.get_parameter('opponent_follow_weight').value),
        )
        self.config.opponent_follow_distance = max(
            0.0,
            float(self.get_parameter('opponent_follow_distance').value),
        )
        self.config.opponent_same_lane_width = max(
            0.01,
            float(self.get_parameter('opponent_same_lane_width').value),
        )
        self.config.opponent_pass_weight = max(
            0.0,
            float(self.get_parameter('opponent_pass_weight').value),
        )
        self.config.opponent_pass_lateral_offset = max(
            0.0,
            float(self.get_parameter('opponent_pass_lateral_offset').value),
        )
        self.config.opponent_pass_longitudinal_window = max(
            0.01,
            float(self.get_parameter('opponent_pass_longitudinal_window').value),
        )
        self.config.opponent_auto_wall_check_enabled = bool(
            self.get_parameter('opponent_auto_wall_check_enabled').value
        )
        self.config.opponent_auto_min_wall_clearance = max(
            0.0,
            float(self.get_parameter('opponent_auto_min_wall_clearance').value),
        )
        self.config.opponent_auto_check_steps = max(
            1,
            int(self.get_parameter('opponent_auto_check_steps').value),
        )
        self.config.opponent_auto_min_closing_speed = max(
            0.0,
            float(self.get_parameter('opponent_auto_min_closing_speed').value),
        )
        self.config.opponent_auto_max_ahead_distance = max(
            0.0,
            float(self.get_parameter('opponent_auto_max_ahead_distance').value),
        )
        self.config.opponent_auto_side_switch_margin = max(
            0.0,
            float(self.get_parameter('opponent_auto_side_switch_margin').value),
        )
        # Auto-derive exit thresholds from enter thresholds when unset (=0).
        # Looser exits = hysteresis: a transient dip during a committed pass
        # won't abort it.
        exit_clear_param = float(self.get_parameter('opponent_auto_exit_wall_clearance').value)
        self.config.opponent_auto_exit_wall_clearance = (
            exit_clear_param if exit_clear_param > 1e-6
            else 0.7 * self.config.opponent_auto_min_wall_clearance
        )
        exit_closing_param = float(self.get_parameter('opponent_auto_exit_closing_speed').value)
        self.config.opponent_auto_exit_closing_speed = (
            exit_closing_param if exit_closing_param > 1e-6
            else 0.4 * self.config.opponent_auto_min_closing_speed
        )
        self.config.opponent_auto_min_commit_sec = max(
            0.0,
            float(self.get_parameter('opponent_auto_min_commit_sec').value),
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
        self.config.mppi_guard_on_timing_jump = bool(
            self.get_parameter('mppi_guard_on_timing_jump').value
        )
        self.config.mppi_guard_wall_gap = max(
            0.0,
            float(self.get_parameter('mppi_guard_wall_gap').value),
        )
        self.config.mppi_guard_stamp_gap = max(
            0.0,
            float(self.get_parameter('mppi_guard_stamp_gap').value),
        )
        self.config.mppi_guard_hard_gap = max(
            self.config.mppi_guard_stamp_gap,
            float(self.get_parameter('mppi_guard_hard_gap').value),
        )
        self.config.mppi_guard_aopt_threshold = float(np.clip(
            float(self.get_parameter('mppi_guard_aopt_threshold').value),
            0.0,
            1.0,
        ))
        self.config.mppi_guard_saturation_callbacks = max(
            1,
            int(self.get_parameter('mppi_guard_saturation_callbacks').value),
        )
        self.config.mppi_guard_bad_callbacks_to_clear_control = max(
            1,
            int(self.get_parameter('mppi_guard_bad_callbacks_to_clear_control').value),
        )
        self.config.state_est_vy_prior = float(np.clip(
            float(self.get_parameter('state_est_vy_prior').value),
            0.0,
            0.95,
        ))
        self.config.state_est_wz_prior = float(np.clip(
            float(self.get_parameter('state_est_wz_prior').value),
            0.0,
            0.95,
        ))
        self.config.state_est_hiccup_dt = max(
            0.0,
            float(self.get_parameter('state_est_hiccup_dt').value),
        )
        self.config.state_est_hiccup_prior_scale = float(np.clip(
            float(self.get_parameter('state_est_hiccup_prior_scale').value),
            0.0,
            1.0,
        ))
        if startup:
            self.config.control_loop_hz = float(np.clip(
                float(self.get_parameter('control_loop_hz').value), 1.0, 100.0,
            ))
        self.config.control_pose_stale_sec = max(
            0.0, float(self.get_parameter('control_pose_stale_sec').value),
        )
        self.config.stats_log_interval_sec = max(
            0.0, float(self.get_parameter('stats_log_interval_sec').value),
        )

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

    def now_sec(self):
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def opponent_path_callback(self, msg):
        self._stats_opponent_rx += 1
        poses = list(msg.poses)
        if not poses:
            self.opponent_path_time = None
            self.opponent_xy_horizon = np.zeros((self.config.n_steps, 2), dtype=np.float32)
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
        self.opponent_path_time = self.now_sec()

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

        age = self.now_sec() - self.opponent_path_time
        active = (
            self.config.opponent_cost_enabled
            and age <= self.config.opponent_path_timeout
            and np.isfinite(horizon).all()
        )
        return horizon, active, age

    def opponent_horizon_speed(self, opponent_traj):
        opponent_traj = np.asarray(opponent_traj, dtype=float)
        if opponent_traj.ndim != 2 or opponent_traj.shape[0] < 2:
            return 0.0
        dt = max(1e-3, float(self.config.sim_time_step))
        step = opponent_traj[1] - opponent_traj[0]
        speed = float(np.linalg.norm(step) / dt)
        if not np.isfinite(speed):
            return 0.0
        return speed

    def opponent_pass_clearance(self, opponent_traj, reference_traj, side):
        if (
            self.config.opponent_auto_wall_check_enabled
            and getattr(self.infer_env, 'wall_sdf', None) is None
        ):
            return 0.0

        n = min(
            max(1, int(self.config.opponent_auto_check_steps)),
            opponent_traj.shape[0],
            max(1, reference_traj.shape[0] - 1),
        )
        pass_offset = max(
            float(self.config.opponent_pass_lateral_offset),
            float(self.config.opponent_cost_radius),
        )
        candidates = []
        for k in range(n):
            ref_idx = min(k + 1, reference_traj.shape[0] - 1)
            yaw = float(reference_traj[ref_idx, 3])
            normal = np.asarray([-np.sin(yaw), np.cos(yaw)], dtype=float)
            candidates.append(opponent_traj[k, :2] + side * pass_offset * normal)
        candidates = np.asarray(candidates, dtype=np.float32)

        wall_dist = np.asarray(
            self.infer_env.sample_wall_distance(jnp.asarray(candidates)),
            dtype=float,
        )
        if wall_dist.size == 0 or not np.isfinite(wall_dist).any():
            return 0.0
        return float(np.nanmin(wall_dist))

    def apply_auto_opponent_behavior(self, state_c_0, reference_traj, opponent_traj, opponent_active):
        """Auto-overtake state machine.

        States: idle | follow | pass_left | pass_right.

        Hysteresis: enter thresholds (decide to start a pass) are stricter
        than exit thresholds (decide to abort one). A minimum commit time
        keeps a chosen side locked in for at least opponent_auto_min_commit_sec
        so a one-frame dip in clearance / closing speed doesn't cancel an
        in-progress overtake.

        Mode IDs (consumed by the cost function):
          0 = follow, 1 = clear (no opp cost), 2 = pass_left, 3 = pass_right.
        """
        # Default debug snapshot for stats.
        self.opponent_auto_debug = {
            'mode_id': float(self.config.opponent_behavior_mode_id),
            'left_clearance': -1.0,
            'right_clearance': -1.0,
            'closing_speed': 0.0,
            'pass_allowed': 0.0,
        }
        if self.config.opponent_behavior_mode != 'auto':
            return

        # Helper: transition to a new state, stamp the entry time.
        def goto(new_state, mode_id):
            if self.auto_state != new_state:
                self.auto_state = new_state
                self.auto_state_entered_at = self.now_sec()
            self.config.opponent_behavior_mode_id = mode_id
            self.opponent_auto_debug['mode_id'] = float(mode_id)

        # No opponent at all -> reset to idle, no opp cost active.
        if not opponent_active:
            goto('idle', 1.0)
            return

        # Geometry vs ego.
        ego_xy = np.asarray(state_c_0[:2], dtype=float)
        ego_speed = float(state_c_0[3])
        opp_xy = np.asarray(opponent_traj[0, :2], dtype=float)
        ref_yaw = float(reference_traj[0, 3])
        tangent = np.asarray([np.cos(ref_yaw), np.sin(ref_yaw)], dtype=float)
        rel = opp_xy - ego_xy
        ahead_distance = float(np.dot(rel, tangent))

        opponent_speed = self.opponent_horizon_speed(opponent_traj)
        desired_speed = max(ego_speed, float(reference_traj[0, 2]))
        closing_speed = desired_speed - opponent_speed
        self.opponent_auto_debug['closing_speed'] = float(closing_speed)

        opponent_relevant = (
            ahead_distance > -0.25
            and ahead_distance <= self.config.opponent_auto_max_ahead_distance
        )
        # Opp behind us or out of range -> reset to idle (NOT follow), so we
        # exit any in-progress pass cleanly.
        if not opponent_relevant:
            goto('idle', 1.0)
            return

        # Clearances. opponent_pass_clearance returns 0 when wall_sdf is
        # missing; warn once and treat both sides as "wall info unavailable"
        # (= clearance unconstrained) instead of silently never passing.
        wall_sdf_present = getattr(self.infer_env, 'wall_sdf', None) is not None
        if not wall_sdf_present and not self._auto_warned_no_wall_sdf:
            self.get_logger().warn(
                "Auto-overtake: wall_sdf not loaded -> wall-clearance gating disabled. "
                "Set wall_cost_enabled=true and a valid wall_cost_map_yaml to re-enable."
            )
            self._auto_warned_no_wall_sdf = True
        if wall_sdf_present:
            left_clear = self.opponent_pass_clearance(opponent_traj, reference_traj, side=1.0)
            right_clear = self.opponent_pass_clearance(opponent_traj, reference_traj, side=-1.0)
        else:
            # Sentinel: both sides "clear" but mark debug to make it visible.
            left_clear = right_clear = 999.0
        self.opponent_auto_debug['left_clearance'] = left_clear
        self.opponent_auto_debug['right_clearance'] = right_clear

        # Thresholds.
        enter_clear = float(self.config.opponent_auto_min_wall_clearance)
        exit_clear = float(self.config.opponent_auto_exit_wall_clearance)
        enter_closing = float(self.config.opponent_auto_min_closing_speed)
        exit_closing = float(self.config.opponent_auto_exit_closing_speed)
        side_margin = float(self.config.opponent_auto_side_switch_margin)
        min_commit = float(self.config.opponent_auto_min_commit_sec)

        # ---- If we're already committed to a pass, decide whether to stay. ----
        if self.auto_state in ('pass_left', 'pass_right'):
            side_clear = left_clear if self.auto_state == 'pass_left' else right_clear
            side_id = 2.0 if self.auto_state == 'pass_left' else 3.0
            time_in_state = self.now_sec() - (self.auto_state_entered_at or self.now_sec())
            self.opponent_auto_debug['pass_allowed'] = 1.0

            # Below min commit time: lock in regardless of transient dips.
            if time_in_state < min_commit:
                goto(self.auto_state, side_id)
                return
            # After min commit: abort only if EXIT thresholds are violated.
            committed_pass_failed = (
                side_clear < exit_clear or closing_speed < exit_closing
            )
            if not committed_pass_failed:
                goto(self.auto_state, side_id)
                return
            # Pass failed mid-commit -> revert to follow (don't go straight to
            # idle; we're still close to opp and follow keeps a safe gap).
            self.get_logger().info(
                f"Auto-overtake: aborting {self.auto_state} "
                f"(side_clear={side_clear:.2f}, closing={closing_speed:.2f})"
            )
            goto('follow', 0.0)
            # Fall through to engage check (could re-engage other side).

        # ---- State is idle/follow: decide whether to engage a new pass. ----
        if closing_speed < enter_closing:
            goto('follow', 0.0)
            return

        left_ok = left_clear >= enter_clear
        right_ok = right_clear >= enter_clear
        if not left_ok and not right_ok:
            goto('follow', 0.0)
            return

        # Both/either OK -> commit. side_margin gives left a small preference
        # only when both sides are similarly clear (avoids flipping in noise).
        if left_ok and (not right_ok or left_clear >= right_clear + side_margin):
            goto('pass_left', 2.0)
        else:
            goto('pass_right', 3.0)
        self.opponent_auto_debug['pass_allowed'] = 1.0

    def reset_state_estimator(self):
        self.prev_pose_time = None
        self.prev_pose_xy = None
        self.prev_pose_yaw = None
        self.state_est_vx = None
        self.state_est_vy = 0.0
        self.state_est_wz = 0.0

    def clear_persistent_mppi_state(self, reason, reset_state_estimator=True, clear_control=False):
        if hasattr(self, 'mppi'):
            self.mppi.a_opt = jnp.zeros_like(self.mppi.a_opt)
            if self.mppi.a_cov is not None:
                self.mppi.a_cov = self.mppi.a_cov_init
        if reset_state_estimator:
            self.reset_state_estimator()
        if clear_control:
            self.control = np.array([0.0, max(self.config.startup_speed, self.config.min_speed)])
        self.mppi_guard_count += 1
        self.get_logger().warn(f"Cleared MPPI persistent state ({self.mppi_guard_count}): {reason}")

    def handle_bad_callback(self, reason):
        self.mppi_bad_output_count += 1
        clear_control = (
            self.mppi_bad_output_count >= self.config.mppi_guard_bad_callbacks_to_clear_control
        )
        self.clear_persistent_mppi_state(reason, clear_control=clear_control)
        if clear_control:
            self.get_logger().error(
                f"Falling back to startup command after {self.mppi_bad_output_count} bad callbacks."
            )

    def maybe_guard_for_timing(self, wall_dt, stamp_dt):
        """Two-tier guard.

        HARD events (gap >= mppi_guard_hard_gap, or non-monotonic stamps)
        wipe the warm-start — the optimizer's previous plan is too stale to
        be safely shifted forward. Examples: 4 s process freeze, JAX
        recompile, multi-second disk I/O spike.

        SOFT events (gap above the legacy wall_gap/stamp_gap thresholds but
        below the hard threshold) just log + count. We KEEP the warm-start
        because for ~0.3-1.5 s gaps the prior plan is still mostly valid,
        and zeroing a_opt forces several cold solves which feel like a
        steering lag at high speed.
        """
        if not getattr(self.config, 'mppi_guard_on_timing_jump', True):
            return
        hard_gap = self.config.mppi_guard_hard_gap
        # HARD: definitely reset.
        if wall_dt > hard_gap:
            self.clear_persistent_mppi_state(f"HARD wall callback gap {wall_dt:.3f}s")
            return
        if stamp_dt <= 0.0:
            self.clear_persistent_mppi_state(f"non-monotonic odom stamp dt {stamp_dt:.3f}s")
            return
        if stamp_dt > hard_gap:
            self.clear_persistent_mppi_state(f"HARD odom stamp gap {stamp_dt:.3f}s")
            return
        # SOFT: log throttled, keep warm-start. Counted in mppi_soft_guard_count
        # so stats_timer can show it without flooding the terminal.
        if wall_dt > self.config.mppi_guard_wall_gap or \
                (stamp_dt != 0.0 and stamp_dt > self.config.mppi_guard_stamp_gap):
            self.mppi_soft_guard_count += 1
            self.get_logger().warn(
                f"SOFT timing gap (warm-start kept): wall={wall_dt:.3f}s stamp={stamp_dt:.3f}s",
                throttle_duration_sec=1.0,
            )

    def sanitize_opponent_horizon(self):
        horizon = np.asarray(self.opponent_xy_horizon, dtype=np.float32)
        if horizon.shape != (self.config.n_steps, 2) or not np.isfinite(horizon).all():
            self.opponent_xy_horizon = np.zeros((self.config.n_steps, 2), dtype=np.float32)
            self.opponent_path_time = None
            return False
        return True

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
            self.last_timing_debug['prev_pose_dt'] = 0.0
            self.last_timing_debug['state_est_vy'] = raw_vy
            self.last_timing_debug['state_est_wz'] = raw_wz
            return seed_speed, raw_vy, raw_wz, np.arctan2(raw_vy, max(seed_speed, 1e-6))

        dt = callback_time - self.prev_pose_time
        if not np.isfinite(dt) or dt <= 1e-3 or dt > 0.5:
            self.prev_pose_time = callback_time
            self.prev_pose_xy = pose_xy
            self.prev_pose_yaw = theta
            self.last_timing_debug['prev_pose_dt'] = float(dt) if np.isfinite(dt) else -1.0
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

        self.last_timing_debug['prev_pose_dt'] = float(dt)
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

        prior_scale = 1.0
        if self.config.state_est_hiccup_dt > 0.0 and dt > self.config.state_est_hiccup_dt:
            prior_scale = self.config.state_est_hiccup_prior_scale

        vy_prior = float(np.clip(self.config.state_est_vy_prior * prior_scale, 0.0, 0.95))
        wz_prior = float(np.clip(self.config.state_est_wz_prior * prior_scale, 0.0, 0.95))

        vy_obs = 0.8 * vy_pose + 0.2 * raw_vy
        vy_est = (1.0 - vy_prior) * vy_obs + vy_prior * self.state_est_vy
        vy_est = float(np.clip(vy_est, -2.0, 2.0))

        wz_obs = 0.85 * wz_pose + 0.15 * raw_wz
        wz_est = (1.0 - wz_prior) * wz_obs + wz_prior * self.state_est_wz
        wz_est = float(np.clip(wz_est, -6.0, 6.0))

        self.prev_pose_time = callback_time
        self.prev_pose_xy = pose_xy
        self.prev_pose_yaw = theta
        self.state_est_vx = vx_est
        self.state_est_vy = vy_est
        self.state_est_wz = wz_est
        self.last_timing_debug['state_est_vy'] = vy_est
        self.last_timing_debug['state_est_wz'] = wz_est

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
            traj_for_viz = self.viz_traj_opt if self.viz_traj_opt is not None else self.mppi.traj_opt
            opt_traj_xy = self.mppi_state_trajectory_to_xy(numpify(traj_for_viz))
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
            self.config.opponent_behavior_mode_id,
            self.config.opponent_follow_weight,
            self.config.opponent_follow_distance,
            self.config.opponent_same_lane_width,
            self.config.opponent_pass_weight,
            self.config.opponent_pass_lateral_offset,
            self.config.opponent_pass_longitudinal_window,
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
        debug_terms['opponent_auto_mode_id'] = self.opponent_auto_debug.get('mode_id', 0.0)
        debug_terms['opponent_auto_left_clearance'] = self.opponent_auto_debug.get('left_clearance', -1.0)
        debug_terms['opponent_auto_right_clearance'] = self.opponent_auto_debug.get('right_clearance', -1.0)
        debug_terms['opponent_auto_closing_speed'] = self.opponent_auto_debug.get('closing_speed', 0.0)
        debug_terms['opponent_auto_pass_allowed'] = self.opponent_auto_debug.get('pass_allowed', 0.0)
        debug_terms['callback_wall_dt'] = self.last_timing_debug.get('callback_wall_dt', 0.0)
        debug_terms['callback_stamp_dt'] = self.last_timing_debug.get('callback_stamp_dt', 0.0)
        debug_terms['prev_pose_dt'] = self.last_timing_debug.get('prev_pose_dt', 0.0)
        debug_terms['state_est_vy'] = self.last_timing_debug.get('state_est_vy', 0.0)
        debug_terms['state_est_wz'] = self.last_timing_debug.get('state_est_wz', 0.0)
        debug_terms['mppi_solve_time'] = self.last_timing_debug.get('mppi_solve_time', 0.0)
        debug_terms['mppi_aopt_max_abs'] = self.last_timing_debug.get('mppi_aopt_max_abs', 0.0)
        debug_terms['mppi_saturation_count'] = float(self.mppi_saturation_count)
        debug_terms['mppi_bad_output_count'] = float(self.mppi_bad_output_count)
        debug_terms['mppi_guard_count'] = float(self.mppi_guard_count)

        for key, pub in self.reward_debug_pubs.items():
            if pub.get_subscription_count() == 0:
                continue
            msg = Float32()
            msg.data = float(debug_terms.get(key, 0.0))
            pub.publish(msg)

    def pose_callback(self, pose_msg):
        """
        Lightweight: cache latest odom + recv time. The control loop runs on
        a fixed-rate timer, not on this callback. Keeping this minimal lets
        the cache update during an in-progress MPPI solve under a
        MultiThreadedExecutor — so PF cadence variation does not throttle
        the controller.
        """
        self.latest_pose_msg = pose_msg
        self.latest_pose_recv_time = time.time()
        self._stats_pose_rx += 1

    def control_timer(self):
        """Fixed-rate driver of the MPPI solve.

        Reads the cached pose, checks staleness, and either calls
        control_step or skips. Skipping is silent in /drive (no republish);
        downstream sees the last command persist until a fresh pose lets us
        compute a new one.
        """
        self._stats_control_ticks += 1
        pose_msg = self.latest_pose_msg
        if pose_msg is None:
            self._stats_control_skips_no_pose += 1
            return
        now = time.time()
        msg_stamp = self.stamp_to_sec(pose_msg.header.stamp)
        if msg_stamp <= 0.0 or not np.isfinite(msg_stamp):
            # Fall back to recv time if upstream stamps are bad.
            msg_stamp = self.latest_pose_recv_time or now
        # Pose age uses recv time + (now - recv) as an upper bound on
        # in-pipeline staleness; this is independent of clock drift between
        # PF and the controller.
        recv_age = now - (self.latest_pose_recv_time or now)
        self._stats_pose_age = recv_age
        if recv_age > self.config.control_pose_stale_sec:
            self._stats_control_skips_stale += 1
            return
        self.control_step(pose_msg)

    def control_step(self, pose_msg):
        """The MPPI solve + /drive publish. Was previously `pose_callback`.

        Live params are refreshed by params_refresh_timer (2 Hz), NOT here:
        ~50 mutex'd parameter reads at 40 Hz held the GIL long enough to
        starve pose_callback under the MultiThreadedExecutor. 0.5 s refresh
        latency is fine for interactive `ros2 param set` tuning.
        """
        t1 = time.time()
        pose = pose_msg.pose.pose
        twist = pose_msg.twist.twist
        theta = self.quaternion_to_yaw(pose.orientation)
        callback_time = self.stamp_to_sec(pose_msg.header.stamp)
        if callback_time <= 0.0:
            callback_time = t1
        if not np.isfinite(callback_time):
            return
        pose_values = np.asarray([
            pose.position.x,
            pose.position.y,
            theta,
            twist.linear.x,
            twist.linear.y,
            twist.angular.z,
        ], dtype=float)
        if not np.isfinite(pose_values).all():
            self.handle_bad_callback("non-finite odom input")
            return
        wall_dt = 0.0 if self.last_pose_callback_wall_time is None else t1 - self.last_pose_callback_wall_time
        stamp_dt = 0.0 if self.last_pose_msg_time is None else callback_time - self.last_pose_msg_time
        self.last_pose_callback_wall_time = t1
        self.last_pose_msg_time = callback_time
        self.last_timing_debug['callback_wall_dt'] = float(wall_dt)
        self.last_timing_debug['callback_stamp_dt'] = float(stamp_dt)
        if wall_dt > 0.0 and stamp_dt != 0.0:
            self.maybe_guard_for_timing(wall_dt, stamp_dt)
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
        if not np.isfinite(state_c_0).all():
            self.handle_bad_callback("non-finite estimated state")
            return
        find_waypoint_vel = max(self.config.ref_vel, vx_state)
        
        reference_traj, waypoint_ind = self.infer_env.get_refernece_traj(state_c_0, find_waypoint_vel, self.config.n_steps)
        if not np.isfinite(reference_traj).all():
            self.handle_bad_callback("non-finite reference trajectory")
            return
        opponent_traj, opponent_active, opponent_age = self.get_opponent_horizon()
        if not self.sanitize_opponent_horizon():
            opponent_traj, opponent_active, opponent_age = self.get_opponent_horizon()
        self.apply_auto_opponent_behavior(state_c_0, reference_traj, opponent_traj, opponent_active)

        ## MPPI call
        solve_t0 = time.time()
        self.mppi.update(
            jnp.asarray(state_c_0),
            jnp.asarray(reference_traj),
            jnp.asarray(opponent_traj),
            opponent_active,
        )
        a_opt_cpu = np.asarray(numpify(self.mppi.a_opt))
        solve_time = time.time() - solve_t0
        self.last_timing_debug['mppi_solve_time'] = float(solve_time)
        aopt_max_abs = float(np.nanmax(np.abs(a_opt_cpu))) if a_opt_cpu.size else 0.0
        if not np.isfinite(aopt_max_abs):
            aopt_max_abs = float('inf')
        self.last_timing_debug['mppi_aopt_max_abs'] = aopt_max_abs
        bad_mppi_output = False
        if (
            not np.isfinite(a_opt_cpu).all()
            or not np.isfinite(numpify(self.mppi.traj_opt)).all()
        ):
            bad_mppi_output = True
            self.handle_bad_callback("non-finite MPPI output")
            a_opt_cpu = np.asarray(numpify(self.mppi.a_opt))
        elif aopt_max_abs >= self.config.mppi_guard_aopt_threshold:
            saturated_frac = float(np.mean(np.abs(a_opt_cpu[:min(3, a_opt_cpu.shape[0]), :]) >= self.config.mppi_guard_aopt_threshold))
            if saturated_frac > 0.8:
                self.mppi_saturation_count += 1
                if self.mppi_saturation_count >= self.config.mppi_guard_saturation_callbacks:
                    self.clear_persistent_mppi_state(
                        f"warm start saturation max={aopt_max_abs:.3f} frac={saturated_frac:.2f}",
                        reset_state_estimator=False,
                    )
                    self.mppi_saturation_count = 0
                    a_opt_cpu = np.asarray(numpify(self.mppi.a_opt))
            else:
                self.mppi_saturation_count = 0
        else:
            self.mppi_saturation_count = 0

        if not bad_mppi_output and np.isfinite(a_opt_cpu).all():
            self.mppi_bad_output_count = 0

        # Visualization-only rollout from raw twist. Decouples the marker
        # from the IIR estimator's noisy (vy, wz, beta), which causes the
        # multimodal cost landscape to flip the optimal sample's trajectory
        # shape between callbacks even when the actual /drive command (a
        # weighted average over samples) is smooth. /opt_traj_arr and the
        # reward-debug topics still consume the controller-truth traj_opt.
        self.viz_traj_opt = None
        if (
            self.config.render
            and self.config.use_pose_delta_state_estimate
            and not bad_mppi_output
        ):
            raw_vx = max(float(twist.linear.x), self.config.init_vel)
            raw_wz = float(twist.angular.z)
            raw_beta = float(np.arctan2(float(twist.linear.y), max(abs(raw_vx), 1e-6)))
            viz_state = state_c_0.copy()
            viz_state[3] = raw_vx
            viz_state[5] = raw_wz
            viz_state[6] = raw_beta
            self.viz_traj_opt = self.mppi.rollout(
                self.mppi.a_opt,
                jnp.asarray(viz_state),
                self.mppi.jrng.new_key(),
                jnp.asarray(self.config.norm_params, dtype=jnp.float32),
                jnp.asarray(self.config.friction, dtype=jnp.float32),
            )

        mppi_control = a_opt_cpu[0] * self.config.norm_params[0, :2]/2
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

        if vx_state < self.config.init_vel:
            startup_speed = np.clip(
                self.config.startup_speed,
                self.config.min_speed,
                self.config.max_speed,
            )
            self.control[1] = float(max(self.control[1], startup_speed))

        if np.isnan(self.control).any() or np.isinf(self.control).any():
            self.control = np.array([0.0, 0.0])
            self.mppi.a_opt = np.zeros_like(self.mppi.a_opt)

        # Publish the safety-critical control command before any optional
        # debug/visualization work. MarkerArray and rosbag serialization can
        # add large timing spikes on hardware; they should never sit between
        # a solved control and /drive.
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = self.control[0]
        drive_msg.drive.speed = self.control[1]
        self.drive_pub.publish(drive_msg)
        self._stats_drive_tx += 1
        solve_dt = time.time() - t1
        self._stats_solve_times.append(solve_dt)
        if len(self._stats_solve_times) > 1024:
            # bound memory under stats_log_interval=0
            self._stats_solve_times = self._stats_solve_times[-256:]

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

        if self.speed_debug_pub.get_subscription_count() > 0:
            speed_debug = np.array([
                vx_state,
                mppi_speed_command,
                profile_speed_command,
                self.control[1],
                self.config.speed_profile_drive_blend,
            ], dtype=np.float32)
            self.speed_debug_pub.publish(to_multiarray_f32(speed_debug))

    def stats_timer(self):
        """Periodic, NON-buffered Hz/timing report.

        Replaces the old `print(f"MPPI Hz: ...")` block, which was inside the
        callback and used `print()` — block-buffered under `ros2 launch` so it
        only flushed at SIGINT. This uses get_logger() which is unbuffered, and
        runs from a timer so it fires whether or not the callback is running.
        """
        if self.config.stats_log_interval_sec <= 0.0:
            return
        now = time.time()
        window = now - self._stats_window_start
        if window <= 0.0:
            return
        guard_delta = self.mppi_guard_count - self._stats_guard_count_at_window_start
        soft_guard_delta = self.mppi_soft_guard_count - self._stats_soft_guard_count_at_window_start
        solves = self._stats_solve_times
        if solves:
            arr = np.asarray(solves, dtype=float)
            solve_mean = float(arr.mean())
            solve_p99 = float(np.percentile(arr, 99))
            solve_max = float(arr.max())
        else:
            solve_mean = solve_p99 = solve_max = 0.0
        self.get_logger().info(
            "MPPI {:5.1f}Hz | pose_rx {:5.1f}Hz | drive_tx {:5.1f}Hz | "
            "ctrl_ticks {:5.1f}Hz (skip stale={} no_pose={}) | "
            "solve mean={:5.1f}ms p99={:5.1f}ms max={:5.1f}ms | "
            "pose_age={:5.1f}ms | guard+={} (soft+={}) | opp_rx {:4.1f}Hz".format(
                self._stats_drive_tx / window,
                self._stats_pose_rx / window,
                self._stats_drive_tx / window,
                self._stats_control_ticks / window,
                self._stats_control_skips_stale,
                self._stats_control_skips_no_pose,
                solve_mean * 1000.0,
                solve_p99 * 1000.0,
                solve_max * 1000.0,
                self._stats_pose_age * 1000.0,
                guard_delta,
                soft_guard_delta,
                self._stats_opponent_rx / window,
            )
        )
        # Reset window.
        self._stats_window_start = now
        self._stats_pose_rx = 0
        self._stats_drive_tx = 0
        self._stats_control_ticks = 0
        self._stats_control_skips_stale = 0
        self._stats_control_skips_no_pose = 0
        self._stats_solve_times = []
        self._stats_guard_count_at_window_start = self.mppi_guard_count
        self._stats_soft_guard_count_at_window_start = self.mppi_soft_guard_count
        self._stats_opponent_rx = 0


def main(args=None):
    rclpy.init(args=args)
    mppi_node = MPPI_Node()
    # MultiThreadedExecutor lets the lightweight pose_callback (cache-only)
    # interleave with an in-progress MPPI solve. The solve itself releases the
    # GIL during JAX GPU calls and during numpify(), so the cache update
    # actually runs concurrently. With a single-threaded executor, a slow
    # solve would block all callbacks and PF cadence variation would directly
    # throttle the controller.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(mppi_node)
    try:
        executor.spin()
    finally:
        mppi_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
