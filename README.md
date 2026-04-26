# Opponent-Aware MPPI for F1TENTH Racing

ROS 2 packages running a JAX-based MPPI controller for aggresive performance and opponent prediction + overtaking on F1TENTH cars (sim + real). Built off of [mlab-upenn/mppi_example](https://github.com/mlab-upenn/mppi_example) (MIT, © 2025 xLab for Safe Autonomous Systems).

**Project focus:** high-speed raceline tracking with MPPI, extended toward
opponent-aware prediction, avoidance, and overtaking behavior for F1TENTH
racing.

![MPPI sim run on Skirk](media/mppi_sim_skirk.gif)

## What it does

At every control step the planner samples **N = 2048** control sequences
$U^{(i)} = u_0^{(i)}, \dots, u_{H-1}^{(i)}$ around a nominal sequence
$\bar U$, rolls each one through a JAX-vmapped vehicle model over a horizon
$H$, and scores the resulting trajectories. The score combines tracking
rewards (xy, yaw, velocity vs. raceline) with cost terms for wall proximity,
sideslip $\beta$, lateral acceleration, and steering saturation — each
toggleable from the YAML.

Sample weights use the standard MPPI softmax with temperature $\lambda$:

$$w^{(i)} = \frac{\exp(R^{(i)} / \lambda)}{\sum_j \exp(R^{(j)} / \lambda)}, \qquad
\bar U \leftarrow \sum_i w^{(i)} U^{(i)}$$

Low $\lambda$ (≈0.01) makes the average winner-take-all; higher $\lambda$
smooths it. Only the first action of $\bar U$ is published to `/drive`; the
rest of $\bar U$ is **shifted by one step and reused as the prior** for the
next solve, so each iteration warm-starts from the last solution instead of
restarting from zero.

The reference trajectory (waypoints + per-waypoint target speeds) can be
swapped **live** via the `/mppi/update_raceline` ROS 2 service — no node
restart. This is driven from the
[raceline_UI_f1tenth](https://github.com/cedrichld/raceline_UI_f1tenth)
web app, which lets you edit racelines and push them to the running
controller on the fly.

## Layout

- `mppi_example/` — controller. `mppi_node.py` (odom → `/drive`),
  `mppi_tracking.py` (JAX rollout loop), `dynamics_models/` (vehicle models).
- `mppi_bringup/` — launch files, params, waypoint CSVs (Levine 9-column
  format under `waypoints/sim/`).
- `opponent_predictor/` — raceline-progress opponent prediction from odometry,
  with Foxglove/RViz visualization and debug topics.
- `MPPI_GUIDE.md` — math, code map, tuning notes.
- `CBF_MPPI_README.md` — earlier notes on CBFs; not the current primary
  project direction.

## Build

```bash
cd ~/ros2_ws/roboracer_ws
colcon build --packages-select mppi_example mppi_bringup opponent_predictor
source install/setup.bash
```

## Run (sim)

```bash
ros2 launch mppi_bringup mppi.launch.py
```

The launch publishes 20 zero-drive messages to keep the car still while JAX
warms up, then starts `lmppi_node` after a 1.6 s timer. Override with
`params_file:=...` or `drive_topic:=...`.

## Changes vs upstream

- `mppi_bringup` package: bundled launch + a single `params_sim.yaml`.
- `mppi_node.py`: `wpt_path` / `wpt_path_absolute` parameters that load a
  raceline directly from CSV, bypassing the upstream `map_info.txt` flow.
- `Track.load_map_from_csv()`: standalone loader for the same.
- Reward weights (`xy_reward_weight`, `velocity_reward_weight`,
  `yaw_reward_weight`) and RViz marker publishers
  (`/mppi/reference`, `/mppi/optimal_trajectory`,
  `/mppi/sampled_trajectories`) added to the node.

## License

MIT — see [`LICENSE`](LICENSE). Includes upstream's MIT notice from xLab.
