# MPPI: Math, Code, and Racing Guide

---

## Part 1: The Math

### 1.1 The Problem

We want to find a control sequence $\mathbf{u} = [u_0, u_1, \dots, u_{H-1}]$ that minimizes the expected cost over a horizon $H$:

$$
\min_{\mathbf{u}} \ \mathbb{E}\left[ \sum_{t=0}^{H-1} \ell(x_t, u_t) + \ell_H(x_H) \right]
$$

subject to:

$$
x_{t+1} = f(x_t, u_t) + w_t, \quad w_t \sim \mathcal{N}(0, \Sigma_w)
$$

where:
- $x_t$ = state at time $t$ (position, velocity, heading, etc.)
- $u_t$ = control at time $t$ (steering velocity, acceleration)
- $\ell(x, u)$ = stage cost (how bad is this state+control?)
- $\ell_H(x_H)$ = terminal cost (how bad is the final state?)
- $f(x, u)$ = dynamics model (bicycle, single-track, etc.)
- $w_t$ = process noise (model uncertainty)

MPC solves this with a gradient-based solver (QP/NLP). MPPI solves it by **sampling**.

**Implementation note:** the code in `mppi_example` does not inject random process noise into the state dynamics. The dynamics rollout is deterministic; the randomness is in the sampled **control perturbations**:

$$
\tilde u_{k,t} = \mathrm{clip}(u_{\mathrm{nom},t} + \varepsilon_{k,t}, -1, 1)
$$

So in this repo, think "many possible control sequences through one dynamics model," not "one control sequence through many noisy worlds."

### 1.2 The Path Integral Insight

Instead of optimizing over controls, use the path integral result (Kappen 2005, Theodorou 2010):

$$
u^*(x, t) = \frac{\mathbb{E}_q\!\left[ u \cdot \exp\!\left(-\tfrac{1}{\lambda} S(\tau)\right) \right]}{\mathbb{E}_q\!\left[ \exp\!\left(-\tfrac{1}{\lambda} S(\tau)\right) \right]}
$$

This says: **the optimal control is a cost-weighted average over sampled trajectories.** No gradients needed.

- $S(\tau) = \sum_{t=0}^{H-1} \ell(x_t, u_t) + \ell_H(x_H)$ is the total cost of trajectory $\tau$
- $\lambda$ is the temperature parameter
- $\exp(-S/\lambda)$ is the exponential weight (low cost $\Rightarrow$ high weight)
- The denominator normalizes so weights sum to 1

### 1.3 Discretization $\rightarrow$ The MPPI Algorithm

Replace the continuous expectation with $K$ discrete samples.

**Importance weights (softmax):**

$$
w_k = \frac{\exp\!\left(-S_k / \lambda\right)}{\sum_{j=1}^{K} \exp\!\left(-S_j / \lambda\right)} = \mathrm{softmax}(-S_k/\lambda)
$$

This is identical to the softmax function from ML. It converts $K$ cost values into $K$ normalized weights that sum to 1.

- $S_k$ = total cost of the $k$-th sampled trajectory
- Low cost $\Rightarrow$ high weight (the negative sign flips it)
- $\lambda$ controls sharpness:
  - $\lambda \to 0$: only the best sample matters ($w_k \approx 1$ for $k = \arg\min S_k$)
  - $\lambda \to \infty$: all samples weighted equally ($w_k = 1/K$)
  - Practical range: $0.01$ to $1.0$

**Weighted control update:**

$$
u_\text{nom} \leftarrow u_\text{nom} + \sum_{k=1}^{K} w_k \, \varepsilon_k
$$

Note: we average the **noise perturbations** $\varepsilon_k$, not the absolute controls. The update nudges the nominal control toward directions that produced low-cost trajectories.

**Cost vs reward in this code:** the equations above use cost $S$, where lower is better. The actual implementation computes rewards that are negative costs, so **higher is better**. That is why the code uses `exp(R_stdzd / temperature)` after normalizing reward-to-go values, instead of explicitly computing `exp(-S / temperature)`.

### 1.4 The Full MPPI Step (Slides 30-31)

Given current state $x_0$ and nominal control sequence $u_\text{nom} \in \mathbb{R}^{H \times 2}$:

**Step 1 — Sample noise:**
$$
\varepsilon_k \sim \mathcal{N}(0, \Sigma), \quad k = 1, \dots, K
$$
Each $\varepsilon_k \in \mathbb{R}^{H \times 2}$ is a full sequence of perturbations.

**Step 2 — Perturb nominal:**
$$
\tilde u_k = \mathrm{clip}(u_\text{nom} + \varepsilon_k, -1, 1)
$$

**Step 3 — Rollout each perturbed control:**
$$
\tau_k: x_0 \xrightarrow{\tilde u_{k,0}} x_1^{(k)} \xrightarrow{\tilde u_{k,1}} \cdots \xrightarrow{\tilde u_{k,H-1}} x_H^{(k)}
$$

**Step 4 — Compute trajectory scores:**
$$
S_k = \sum_{t=0}^{H-1} \ell\!\left(x_t^{(k)}, \tilde u_{k,t}\right) + \ell_H\!\left(x_H^{(k)}\right)
$$

In the repo this is represented as a reward vector, roughly $r_t = -\ell(x_t, u_t)$, then accumulated into reward-to-go values.

**Step 5 — Compute weights and update nominal:**
$$
w_k = \mathrm{softmax}(-S_k/\lambda), \qquad u_\text{nom} \leftarrow u_\text{nom} + \sum_{k=1}^{K} w_k \, \varepsilon_k
$$

**Step 6 — Execute and shift (warm start):**
Apply $u_\text{nom}[0]$ to the car, then shift the sequence forward:
$$
u_\text{nom} \leftarrow \big[\, u_\text{nom}[1],\, u_\text{nom}[2],\, \dots,\, u_\text{nom}[H-1],\, \mathbf{0} \,\big]
$$

Repeat at the next control timestep.

### 1.5 Numerical Stability

Naive computation of $\exp(-S_k/\lambda)$ with $\lambda = 0.01$ and costs in the hundreds produces numbers like $\exp(-10000)$, which underflows to zero.

For costs, a stable version would use something like:

$$
\tilde S_k = \frac{\min_j S_j - S_k}{(\max_j S_j - \min_j S_j) + \eta}, \qquad w_k = \frac{\exp(\tilde S_k / \lambda)}{\sum_j \exp(\tilde S_j / \lambda)}
$$

This repo uses rewards instead of costs:

$$
\tilde R_k = \frac{R_k - \max_j R_j}{(\max_j R_j - \min_j R_j) + \eta}, \qquad w_k = \frac{\exp(\tilde R_k / \lambda)}{\sum_j \exp(\tilde R_j / \lambda)}
$$

- The best reward gets exponent $0$, so its unnormalized weight is $1$.
- Worse rewards get negative exponents, so their weights shrink.
- Dividing by the range rescales scores before applying temperature.
- The damping term $\eta = 0.001$ prevents division by zero when all scores are identical.

### 1.6 Key Tuning Parameters

| Parameter | Symbol | Effect |
|---|---|---|
| Temperature | $\lambda$ | Lower $=$ more greedy (follows best sample). Higher $=$ more exploratory |
| Num samples | $K$ | More $=$ better solution quality, slower. Diminishing returns past $\sim 1000$ |
| Horizon | $H$ | Longer $=$ sees further ahead, but more expensive and noise accumulates |
| Noise std | $\sigma$ | Controls exploration radius. Per-channel: different for steering vs accel |
| Num iterations | $n_\text{iter}$ | $>1$ $=$ refine the nominal multiple times per timestep |

### 1.7 What You Can Skip from Lecture

- **Slides 22-26 (HJB, exponential transformation, Desai-Zwanzig/Kappen derivation):** Theoretical proof that softmax weighting is optimal. Does not affect implementation.
- **Slide 23 (controlled diffusion assumption):** In practice, noise is added directly to controls.
- **Slides 14-19 (stochastic Bellman, LQG):** Background. MPPI sidesteps all of this.

---

## Part 2: Code Structure

### 2.1 File Map

```
mppi_example/
├── mppi_node.py                    # ROS2 node — entry point, subscribes/publishes
├── mppi_tracking.py                # Core MPPI algorithm — THE important file
├── infer_env.py                    # Dynamics wrapper + reward functions + reference traj
├── config.yaml                     # All tunable parameters
├── dynamics_models/
│   ├── dynamics_models_jax.py      # Vehicle dynamics models (ST, KS, point mass, frenet)
│   ├── mb_model_params.py          # F1TENTH + real car physical parameters
│   └── tire_models.py              # Pacejka tire model
├── utils/
│   ├── Track.py                    # Track/waypoint loading and Frenet conversion
│   ├── jax_utils.py                # JAX RNG wrapper, numpify helper
│   ├── cubic_spline.py             # Cubic spline interpolation for track centerline
│   └── ros_np_multiarray.py        # numpy ↔ ROS Float32MultiArray conversion
└── waypoints/                      # Track waypoint CSV files
```

### 2.2 mppi_tracking.py — Core MPPI (167 lines)

This is the file that implements the 5-step algorithm. Everything else is support.

**`__init__(self, config, env, ...)`** — Sets up:
- `n_samples` ($K$), `n_steps` ($H$), `temperature` ($\lambda$), `a_std` ($\sigma$)
- `a_opt`: nominal control sequence, shape $(H, 2)$, initialized to zeros
- `accum_matrix`: upper-triangular matrix for computing cumulative returns (reward-to-go at each timestep, not just immediate reward)

**`update(self, env_state, reference_traj)`** — The outer loop:
1. Shift previous nominal forward (`shift_prev_opt`)
2. Run `n_iterations` of `iteration_step`
3. Optionally convert states to Frenet frame

**`iteration_step(self, a_opt, a_cov, rng, state, ref)`** — One full MPPI step (jitted):
- Lines 77-82: **Sample** — `truncated_normal(...)` generates shape $(K, H, 2)$ noise
- Line 84: **Perturb** — `a_opt + da`, clipped to $[-1, 1]$
- Lines 85-86: **Rollout** — `vmap(rollout)(actions, state, rng)` → shape $(K, H, 7)$ states
- Lines 90-96: **Reward** — `vmap(reward_fn)(states, ref)` → shape $(K, H)$ per-step rewards
- Line 98: **Returns** — `vmap(returns)(reward)` → cumulative reward-to-go
- Line 99: **Weights** — `vmap(weights)(R)` → importance weights per timestep
- Line 100: **Update** — `average(da, weights)` → weighted noise, add to nominal

**`rollout(self, actions, state, rng)`** — Forward simulation:
- Python for-loop over $H$ steps (unrolled by jit since $H=10$ is small)
- Each step calls `self.env.step(state, action, rng)` → next state
- Returns shape $(H, 7)$ trajectory

**`weights(self, R)`** — Importance weights:
- Normalizes $R$, exponentiates, normalizes to sum to 1
- Applies temperature and damping for stability

**`shift_prev_opt(self, a_opt, a_cov)`** — Warm start:
- Drops first control, appends zeros at end
- Resets covariance if using adaptive covariance

### 2.3 infer_env.py — Environment Wrapper (342 lines)

Bridges dynamics models with MPPI. Two main responsibilities:

**Dynamics (`step` and `update_fn`):**
- Wraps the dynamics model with RK4 integration (not Euler — more accurate)
- `update_fn` integrates the chosen model (`dynamic_ST` or `kinematic_ST`) over `DT`
- Uses `fori_loop` for sub-stepping (`DT/Ddt` steps of 0.1s each)
- Controls are **normalized**: `u * norm_params[0, :2] / 2` maps $[-1, 1]$ to physical units

**Reference trajectory (`get_refernece_traj`):**
- Finds nearest waypoint to current position
- Projects forward along waypoints based on target speed $\times$ DT
- Interpolates between waypoints for smooth reference
- Returns shape $(H+1, 7)$ array: $[x, y, \text{speed}, \text{yaw}, s, 0, 0]$
- In the current helper, the returned speed column is interpolated from waypoint column 5; `target_speed` mainly controls how far ahead the reference points are spaced.
- Has both numpy (for ROS callback) and JAX versions

**Reward functions:**
- `reward_fn_xy`: L1 position tracking cost in cartesian $(x, y)$
- `reward_fn_sey`: L1 tracking cost in Frenet $(s, e_y)$ frame
- Currently only uses position tracking. Velocity and yaw costs are commented out — these are tuning levers you should explore.

### 2.4 dynamics_models_jax.py — Vehicle Models (1031 lines)

Multiple dynamics models, all jitted:

**`vehicle_dynamics_st`** — Dynamic Single-Track (the one used by default):
- State: $[x, y, \delta, v_x, \psi, \omega, \beta]$ — position, steering angle, speed, yaw, yaw rate, slip angle
- Control: $[\dot\delta, a]$ — steering velocity and acceleration (**not** steering angle directly!)
- Uses linear tire model with cornering stiffness $C_{Sf}, C_{Sr}$
- Automatically switches to kinematic model when $|v_x| < 1$ m/s (low-speed stability)
- Applies steering and acceleration constraints (clipping)

**`vehicle_dynamics_ks`** — Kinematic Single-Track:
- Simpler, 5-state: $[x, y, \delta, v, \psi]$
- No tire forces, just geometric bicycle model
- Good for low speeds, not accurate at the limit

**`vehicle_dynamics_st_pacjeka_frenet`** — Dynamic ST in Frenet frame:
- State: $[s, e_y, \delta, v_x, e_\psi, \omega_z, v_y]$ — curvilinear coordinates
- Uses Pacejka tire model (Magic Formula) for nonlinear tire forces
- More accurate at high slip angles

**Key physical parameters** (from `mb_model_params.py`, F1TENTH):
- Wheelbase: $l_f + l_r = 0.15875 + 0.17145 = 0.3302$ m
- Mass: $3.74$ kg
- Max steering angle: $\pm 0.4189$ rad $(\pm 24°)$
- Max acceleration: $9.51$ m/s$^2$
- Max velocity: $20.0$ m/s

### 2.5 mppi_node.py — ROS2 Node (152 lines)

**Initialization:**
1. Loads config, track, waypoints
2. Creates `InferEnv` (dynamics + reward) and `MPPI` controller
3. Does a dummy MPPI call to trigger JIT compilation before the loop starts
4. Subscribes to odometry, publishes drive commands

**`pose_callback`** — Main control loop (called on every odom message):
1. Extracts state from odometry: $[x, y, \delta, v, \theta, \omega, \beta]$
2. Gets reference trajectory from waypoints
3. Calls `mppi.update(state, reference)` — the MPPI step
4. Converts MPPI output to steering angle + speed:
   - `mppi.a_opt[0]` gives normalized control $[-1, 1]$
   - Multiply by `norm_params / 2` to get physical units
   - Steering: integrates steering velocity $\times$ dt $+$ current angle
   - Speed: integrates acceleration $\times$ dt $+$ current speed
5. Publishes `AckermannDriveStamped`
6. Tracks Hz (prints every 100 iterations)

**Important:** Line 21 has a commented-out JIT cache:
```python
#jax.config.update("jax_compilation_cache_dir", "/home/nvidia/jax_cache")
```
**Enable this on Jetson** to avoid cold-start recompilation every boot.

### 2.6 config.yaml — Tunable Parameters

```yaml
n_samples: 256          # K — number of trajectory samples
n_steps: 10             # H — horizon length
n_iterations: 1         # MPPI iterations per timestep
control_sample_std:     # σ — noise standard deviation
  - 0.5                 #   steering velocity channel
  - 0.5                 #   acceleration channel
state_predictor: dynamic_ST  # which dynamics model
sim_time_step: 0.1      # dt for dynamics integration
ref_vel: 4.0            # target speed used for lookahead spacing (m/s)
friction: 0.8           # tire-road friction coefficient
init_vel: 2.0           # minimum velocity threshold
# Temperature is set in MPPI constructor: temperature=0.01
# Damping is set in MPPI constructor: damping=0.001
```

---

## Part 3: Roadmap to Understanding

The full call chain for one MPPI decision spans 4 files. This roadmap walks you through it in a deliberate order.

### Step 1: Trace a single MPPI call end-to-end (30 min)

**Start here:** [mppi_node.py:106](mppi_example/mppi_node.py#L106)
```python
self.mppi.update(jnp.asarray(state_c_0), jnp.asarray(reference_traj))
```

**Follow into the MPPI class:**
1. [mppi_tracking.py:49](mppi_example/mppi_tracking.py#L49) — `update()`: first shifts the previous nominal forward (warm start), then calls `iteration_step` once.
2. [mppi_tracking.py:75](mppi_example/mppi_tracking.py#L75) — `iteration_step()`: this IS the 5-step algorithm from Part 1.4. Read it slowly with the algorithm in mind. You should be able to label each line with its step number (1. sample, 2. perturb, 3. rollout, 4. reward, 5. update).
3. Notice the shapes at each stage:
   - `da`: $(K, H, 2)$ — noise
   - `actions`: $(K, H, 2)$ — perturbed controls
   - `states`: $(K, H, 7)$ — rolled-out trajectories
   - `reward`: $(K, H)$ — per-step rewards
   - `R`: $(K, H)$ — cumulative returns (reward-to-go)
   - `w`: $(K, H)$ — importance weights
   - `da_opt`: $(H, 2)$ — weighted average of noise
   - `a_opt`: $(H, 2)$ — updated nominal control

### Step 2: Understand the dynamics pipeline (30 min)

Now you know MPPI calls `rollout` which calls `self.env.step`. Trace what that actually does:

1. [mppi_tracking.py:148](mppi_example/mppi_tracking.py#L148) — `rollout_step` calls `self.env.step(env_state, actions, rng_key)`
2. [infer_env.py:65](mppi_example/infer_env.py#L65) — `InferEnv.step()` is a thin wrapper:
   ```python
   return self.update_fn(x, u * self.norm_params[0, :2] / 2)
   ```
   **Important:** this is where controls get denormalized from $[-1, 1]$ to physical units.
3. [infer_env.py:40-48](mppi_example/infer_env.py#L40-L48) — `update_fn` does RK4 integration with sub-stepping
4. [dynamics_models_jax.py:280-358](mppi_example/dynamics_models/dynamics_models_jax.py#L280-L358) — `vehicle_dynamics_st` computes the actual ODEs: tire forces, yaw rate update, slip angle dynamics

**Key mental model:** Inside MPPI, controls live in $[-1, 1]$. They only become real physical quantities (rad/s, m/s²) at two specific points:
- Inside `InferEnv.step()`: for dynamics simulation during rollouts
- Inside `mppi_node.pose_callback()`: when computing the actual drive command to publish

### Step 3: Understand the reference trajectory (20 min)

The reward function compares rolled-out states against a reference. Where does the reference come from?

1. [mppi_node.py:103](mppi_example/mppi_node.py#L103) — `self.infer_env.get_refernece_traj(state, find_waypoint_vel, n_steps)` is called every control cycle
2. [infer_env.py:172](mppi_example/infer_env.py#L172) — `get_refernece_traj`:
   - Finds nearest waypoint via projection (`nearest_point`)
   - Walks forward along waypoints by `speed × DT` per step
   - Interpolates positions, yaw, speed between waypoints
   - Returns shape $(H+1, 7)$: $[x, y, \text{speed}, \text{yaw}, s, 0, 0]$

This is what the reward function tracks against. **Replacing this with your raceline optimizer output is the highest-leverage change you can make** (see Part 4.3).

### Step 4: Understand the reward function (20 min)

Now that you've seen reference generation and rollouts, look at how they're compared.

**Where it's called:** [mppi_tracking.py:89-96](mppi_example/mppi_tracking.py#L89-L96)
```python
if self.config.state_predictor in self.config.cartesian_models:
    reward = jax.vmap(self.env.reward_fn_xy, in_axes=(0, None))(states, reference_traj)
else:
    reward = jax.vmap(self.env.reward_fn_sey, in_axes=(0, None))(states, reference_traj)
```

The `vmap(..., in_axes=(0, None))` means: batch over axis 0 of `states` (the $K$ samples), broadcast the single `reference_traj` to every sample. This is the pattern you learned in JAX: "each of the $K$ trajectories is scored against the same reference."

**What the reward actually computes:** [infer_env.py:86-96](mppi_example/infer_env.py#L86-L96)
```python
def reward_fn_xy(self, state, reference):
    xy_cost = -jnp.linalg.norm(reference[1:, :2] - state[:, :2], ord=1, axis=1)
    vel_cost = -jnp.abs(reference[1:, 2] - state[:, 3])
    yaw_cost = -jnp.abs(jnp.sin(reference[1:, 3]) - jnp.sin(state[:, 4])) - \
        jnp.abs(jnp.cos(reference[1:, 3]) - jnp.cos(state[:, 4]))
    # return 20*xy_cost + 15*vel_cost + 1*yaw_cost
    return xy_cost
```

Observations:
- **Reward, not cost.** Sign is flipped ($-\|\cdot\|$). Bigger reward $=$ better. The code then weights higher reward-to-go values more heavily.
- **Shape:** input `state` is $(H, 7)$, input `reference` is $(H+1, 7)$. Output is $(H,)$ — one reward per timestep along the trajectory.
- **L1 norm** (`ord=1`), not L2. Robust to outliers (single bad step doesn't blow up).
- **Position only** in the active version. The `vel_cost` and `yaw_cost` lines are defined but only used in the commented-out weighted sum. You'll want to uncomment this for racing.
- **`reference[1:, :2]`**: skips index 0 because the reference has $H+1$ points (including current position), while the rollout has $H$ future states. The slice aligns them.

### Step 5: Understand cumulative returns and weights (20 min)

After the reward function produces $(K, H)$ per-step rewards, two more operations happen before the nominal gets updated.

**Returns — reward-to-go, not just immediate reward:** [mppi_tracking.py:119-122](mppi_example/mppi_tracking.py#L119-L122)
```python
@partial(jax.jit, static_argnums=(0))
def returns(self, r):
    return jnp.dot(self.accum_matrix, r)
```

The `accum_matrix` is an upper-triangular ones matrix of shape $(H, H)$:

$$
A = \begin{bmatrix} 1 & 1 & 1 & \cdots & 1 \\ 0 & 1 & 1 & \cdots & 1 \\ 0 & 0 & 1 & \cdots & 1 \\ \vdots & & & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & 1 \end{bmatrix}
$$

Multiplying this by the reward vector $r = [r_0, r_1, \dots, r_{H-1}]$ gives:

$$
R_t = \sum_{s=t}^{H-1} r_s
$$

So $R_t$ is the **reward-to-go from timestep $t$ onwards**, not just the immediate reward. This gives MPPI a time-varying weighting where early controls are evaluated against the whole future, and later controls are evaluated against their remaining horizon. This is a subtle refinement over the naive "one scalar score per trajectory" version you saw in the lecture slides.

**Weights — softmax per timestep:** [mppi_tracking.py:125-133](mppi_example/mppi_tracking.py#L125-L133)
```python
@partial(jax.jit, static_argnums=(0))
def weights(self, R):
    R_stdzd = (R - jnp.max(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)
    w = jnp.exp(R_stdzd / self.temperature)
    w = w / jnp.sum(w)
    return w
```

This is the numerical stability trick from Part 1.5, applied per timestep to rewards. Note the call site at [mppi_tracking.py:99](mppi_example/mppi_tracking.py#L99):
```python
w = jax.vmap(self.weights, 1, 1)(R)
```

`vmap(..., in_axes=1, out_axes=1)` means: batch over **axis 1** (timestep), not axis 0 (sample). So for each timestep $t$, a separate softmax is computed over all $K$ samples. Result: `w` has shape $(K, H)$ — one weight per (sample, timestep) pair.

**The update — weighted average of noise:** [mppi_tracking.py:100-101](mppi_example/mppi_tracking.py#L100-L101)
```python
da_opt = jax.vmap(jnp.average, (1, None, 1))(da, 0, w)
a_opt = jnp.clip(a_opt + da_opt, -1.0, 1.0)
```

`jnp.average(array, axis, weights)` computes a weighted average. The `vmap` here batches over timestep (axis 1 of `da` and axis 1 of `w`), and for each timestep computes:

$$
(da\_opt)_t = \sum_{k=1}^{K} w_{k,t} \cdot (da)_{k, t, :}
$$

So each timestep of the nominal gets its own weighted update based on its own softmax. Then `a_opt + da_opt` nudges the nominal toward high-reward directions, and clipping keeps it in the valid control range.

**Summary of the data flow:**

$$
\underbrace{(K, H)}_{\text{reward}} \xrightarrow{\text{accum\_matrix}} \underbrace{(K, H)}_{R,\ \text{reward-to-go}} \xrightarrow{\text{softmax per } t} \underbrace{(K, H)}_{w} \xrightarrow{\text{wavg over } K} \underbrace{(H, 2)}_{da\_opt} \xrightarrow{+ a\_opt} \underbrace{(H, 2)}_{\text{new nominal}}
$$

---

## Part 4: Performance Improvements & Project Additions

### 4.1 Quick Wins (change config, no code)

1. **Increase `n_samples`**: 256 → 512 or 1024. Better solution quality at cost of Hz.
2. **Tune `temperature`**: 0.01 is very greedy. Try 0.05-0.1 if behavior is jerky.
3. **Tune `control_sample_std`**: Different values for steering vs accel. Steering might need less noise (e.g., 0.3) while accel might need more.
4. **Increase `n_steps`**: 10 → 15 or 20 for longer lookahead. Important at higher speeds.
5. **Adjust `ref_vel`**: this mostly changes how far ahead the reference points are spaced. Increase as the car gets stable, but remember velocity tracking needs a meaningful speed column in the reference.

**Keep in mind from later tuning work:**

- **Keep `init_vel` close to `speed_profile_min_speed`**: if they are far apart, the controller can start in a mismatched regime where the state estimate thinks the car should already be moving at one speed while the reference profile assumes another.
- **In sim, good results can come from physically accurate MPPI even without extra costs**: once friction and horizon are tuned correctly, the controller can run very well with just the core tracking objective. In that regime, `use_speed_profile_drive_speed` and the drive-speed blend can actually make tuning harder by mixing profile feedforward into what should be a clean test of the MPPI physics/model.

### 4.2 Cost Function Improvements (modify infer_env.py)

The current cost only tracks position. For racing, you want:

1. **Enable velocity tracking**: Uncomment `vel_cost` in `reward_fn_xy`. Weight it to encourage speed.
2. **Enable yaw tracking**: Uncomment `yaw_cost`. Helps the car align with the track direction, not just position.
3. **Add obstacle/opponent cost**: Check distance to opponent car, add large penalty for proximity. Since MPPI never differentiates through cost, you can use hard indicator functions.
4. **Add wall/boundary cost**: Penalize proximity to track boundaries using a distance field or occupancy grid.
5. **Add control smoothness cost**: Penalize large changes in steering between timesteps to reduce jitter.

### 4.3 Raceline Integration (modify infer_env.py)

**The highest-leverage change.** Replace raw waypoint reference with your optimized raceline:

Currently `get_refernece_traj` walks along raw waypoints using a target speed for spacing, while the returned speed column comes from the waypoint CSV. Your raceline optimizer produces an optimal racing line with a speed profile. Feed this in instead:
- Replace the waypoints loaded in `InferEnv.__init__` with your raceline CSV
- Or modify `get_refernece_traj` to use raceline positions + speeds instead of the current waypoint speed column
- The reference already has slots for $[x, y, \text{speed}, \text{yaw}, s, 0, 0]$ — your raceline has all of these

This gives you: optimal high-level path (where to go) + MPPI (how to get there with real physics and obstacle avoidance).

### 4.4 Multi-Iteration Refinement (modify config)

Set `n_iterations: 2` or `3`. Each iteration re-samples around the *updated* nominal, progressively tightening the solution. The TA code already supports this. Trade-off: multiplies compute per timestep.

### 4.5 Adaptive Covariance (already in code)

Set `adaptive_covariance: true` and `a_cov_shift: true` in config. This makes the noise covariance adapt based on which samples were good — samples cluster around good trajectories over iterations. Already implemented in [mppi_tracking.py:102-109](mppi_example/mppi_tracking.py#L102-L109).

---

## Part 5: Sim-to-Real Issues

### 5.1 Dynamics Model Mismatch

The biggest risk. The dynamics model in sim and on the real car will behave differently:
- **Tire parameters**: $C_{Sf}, C_{Sr}, \mu$ in `mb_model_params.py` are estimates. Real tires behave differently, especially at the limit. The `friction: 0.8` in config is a global multiplier — tune on real car.
- **Steering dynamics**: Real servo has delay and rate limits. The model assumes instant steering velocity response.
- **Speed**: Encoder noise, slip at high speeds.

**Mitigation**: Start conservative (low speed, high temperature), tune parameters on real car, compare predicted vs actual trajectories.

### 5.2 Latency

MPPI must run fast enough for real-time control. Sources of latency:
- **Computation**: MPPI step takes $X$ ms. Track with the built-in Hz counter. Target $\geq 20$ Hz.
- **Sensor delay**: Odometry arrives with latency. Consider predicting state forward by latency amount before running MPPI.
- **Actuator delay**: Steering servo and ESC have response time. The warm start (shifting nominal) partially accounts for this.

### 5.3 JAX on Jetson

- **Cold start**: First MPPI call triggers XLA compilation ($\sim 30$-$60$ s). Enable jit cache: `jax.config.update("jax_compilation_cache_dir", "/path/to/cache")`
- **Memory**: Jetson has limited GPU memory. `n_samples=2000` might OOM. Monitor with `jtop`.
- **Installation**: JAX wheels for aarch64 (Jetson) are different from x86_64 (your laptop). May need to build from source or find Jetson-specific wheels.
- **GPU vs CPU**: Verify JAX is using GPU on Jetson: `python -c "import jax; print(jax.devices())"`

### 5.4 Control Normalization

Controls in MPPI are normalized to $[-1, 1]$. The denormalization happens in two places:
1. `infer_env.step()`: for dynamics simulation during rollouts
2. `mppi_node.pose_callback()`: for actual drive commands

If `norm_params` doesn't match your car's actual limits, the car will over/under-steer. Verify these match your car's physical capabilities:
```yaml
norm_params:
  - [6.0, -3.0]
  - [8.0, -4.0]
```

The node transposes this array after loading it, then uses only `norm_params[0, :2] / 2`. With the current YAML, normalized controls map approximately to:
- steering velocity: $[-3, 3]$ rad/s
- acceleration: $[-4, 4]$ m/s²

The negative row is not currently used in the denormalization path, so treat this field carefully if you refactor it.

### 5.5 Reference Trajectory Quality

If the reference trajectory is noisy or has discontinuities (e.g., yaw wrapping at $\pm\pi$), MPPI will produce jerky controls. The code already handles yaw wrapping (lines 158-168 in `infer_env.py`), but verify this works with your map.

### 5.6 Warm Start Failure Modes

If something unexpected happens (opponent appears, car gets bumped), the warm-started nominal is stale and may not explore the right region of control space. Symptoms: car keeps trying the old plan. Mitigation: increase temperature temporarily, or add more noise when tracking error is high.
