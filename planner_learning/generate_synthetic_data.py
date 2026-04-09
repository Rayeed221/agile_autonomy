#!/usr/bin/env python3
"""
Synthetic Data Generator for Agile Autonomy Training

Generates physically plausible drone flight data in the exact format expected by
PlanDataset. Uses fully vectorised numpy ray-AABB intersection for speed.

Estimated time: ~3-8 min for 20 rollouts × 200 steps on CPU.
"""

import os, math, argparse
import numpy as np
import pandas as pd
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R_scipy
from tqdm import tqdm

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_W, IMG_H     = 224, 224
RENDER_W         = 64           # render at low res; loader resizes to IMG_W×IMG_H
RENDER_H         = 48
HFOV             = math.radians(90)
VFOV             = math.radians(70)
MAX_DEPTH_MM     = 20000        # 20 m clip
DT               = 1.0 / 15.0  # 15 Hz
AVG_SPEED        = 7.0          # m/s
OUT_SEQ_LEN      = 10
STATE_DIM        = 3
N_TRAJ_CAND      = 10
QUAD_RADIUS      = 0.5          # collision sphere radius (m)


# ─── Vectorised Ray–AABB environment ─────────────────────────────────────────
class Environment:
    """Random forest of axis-aligned box obstacles (all numpy, vectorised)."""

    def __init__(self, n_obstacles=60, seed=42,
                 corridor_len=120.0, corridor_w=15.0,
                 min_size=0.4, max_size=2.5):
        rng = np.random.RandomState(seed)
        cx  = rng.uniform(10, corridor_len - 10, n_obstacles)
        cy  = rng.uniform(-corridor_w/2, corridor_w/2, n_obstacles)
        cz  = rng.uniform(0.3, 1.5, n_obstacles)

        hx  = rng.uniform(min_size/2, max_size/2, n_obstacles)
        hy  = rng.uniform(min_size/2, max_size/2, n_obstacles)
        hz  = rng.uniform(min_size/2, max_size/2, n_obstacles) + cz

        # shapes: (n_obs, 3)
        self.centers  = np.stack([cx, cy, cz],   axis=1).astype(np.float32)
        self.halves   = np.stack([hx, hy, hz],   axis=1).astype(np.float32)
        self.box_min  = self.centers - self.halves   # (n, 3)
        self.box_max  = self.centers + self.halves   # (n, 3)

    # ── Point cloud ──────────────────────────────────────────────────────────
    def point_cloud(self, n_per_box=25):
        rng = np.random.RandomState(0)
        pts = []
        for i in range(len(self.centers)):
            c, h = self.centers[i], self.halves[i]
            for _ in range(n_per_box):
                axis = rng.randint(3)
                sign = rng.choice([-1, 1])
                pt   = c + rng.uniform(-1, 1, 3) * h
                pt[axis] = c[axis] + sign * h[axis]
                pts.append(pt)
        return np.array(pts, dtype=np.float32)

    # ── Batch ray–AABB for all pixels at once ──────────────────────────────
    def render_depth_batch(self, origin, dirs_world):
        """
        origin     : (3,)
        dirs_world : (N, 3) unit vectors
        Returns    : (N,) distances in metres, clipped to 20 m.
        """
        N   = dirs_world.shape[0]
        n   = self.centers.shape[0]

        # Expand for broadcast: origin (1,3), dirs (N,1,3), boxes (1,n,3)
        orig  = origin[None, None, :]         # (1, 1, 3)
        dirs  = dirs_world[:, None, :]        # (N, 1, 3)
        mn    = self.box_min[None, :, :]      # (1, n, 3)
        mx    = self.box_max[None, :, :]      # (1, n, 3)

        inv_d = 1.0 / (dirs + 1e-12)          # (N, 1, 3) broadcast

        t0    = (mn - orig) * inv_d            # (N, n, 3)
        t1    = (mx - orig) * inv_d            # (N, n, 3)

        t_near = np.max(np.minimum(t0, t1), axis=2)   # (N, n)
        t_far  = np.min(np.maximum(t0, t1), axis=2)   # (N, n)

        hit  = (t_near <= t_far) & (t_far > 0)        # (N, n)
        t_hit = np.where(hit, np.maximum(t_near, 0.0), 20.0)

        return t_hit.min(axis=1)              # (N,)  nearest hit per ray

    # ── Render depth image ──────────────────────────────────────────────────
    def render_depth(self, pos, R_bw, width=RENDER_W, height=RENDER_H):
        fx = width  / (2 * math.tan(HFOV / 2))
        fy = height / (2 * math.tan(VFOV / 2))
        u  = np.arange(width);  v = np.arange(height)
        uu, vv = np.meshgrid(u, v)
        xc = (uu - width/2)  / fx
        yc = (vv - height/2) / fy
        zc = np.ones_like(xc)
        dirs_cam   = np.stack([zc, -xc, -yc], axis=-1).reshape(-1, 3)
        dirs_world = dirs_cam @ R_bw.T
        norms      = np.linalg.norm(dirs_world, axis=1, keepdims=True)
        dirs_world = dirs_world / (norms + 1e-9)

        depths = self.render_depth_batch(pos, dirs_world)
        depth_mm = (depths * 1000).clip(0, MAX_DEPTH_MM).astype(np.uint16)
        img = depth_mm.reshape(height, width)
        # Resize to full resolution (loader also resizes but having the right
        # size avoids issues)
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        return img

    # ── Batch collision check for trajectory waypoints ───────────────────────
    def traj_costs(self, waypoints_wf):
        """
        waypoints_wf : (n_cand, n_steps, 3) world-frame positions
        Returns      : (n_cand,) collision costs
        """
        n_c, n_s, _ = waypoints_wf.shape
        pts = waypoints_wf.reshape(-1, 3)          # (n_c*n_s, 3)

        # Closest point on each box to each waypoint
        mn = self.box_min[None, :, :]              # (1, n, 3)
        mx = self.box_max[None, :, :]              # (1, n, 3)
        pt = pts[:, None, :]                       # (M, 1, 3)

        closest = np.clip(pt, mn, mx)              # (M, n, 3)
        dists   = np.linalg.norm(pt - closest, axis=2)  # (M, n)
        min_d   = dists.min(axis=1)                # (M,) closest distance to any box

        # Cost: parabolic inside 0.8m, linear 0.8→QUAD_RADIUS
        cost_per_pt = np.where(
            min_d < QUAD_RADIUS,
            -2.0 / (QUAD_RADIUS**2) * min_d**2 + 4.0,
            np.where(min_d < 0.8,
                     2.0 * (QUAD_RADIUS - min_d) / (0.8 - QUAD_RADIUS) + 2.0,
                     0.0)
        )
        costs = cost_per_pt.reshape(n_c, n_s).mean(axis=1)
        return costs.astype(np.float32)


# ─── Trajectory generation ────────────────────────────────────────────────────
def generate_trajectories_bf(env, pos, R_bw,
                              n_traj=N_TRAJ_CAND, out_steps=OUT_SEQ_LEN,
                              dt=DT, speed=AVG_SPEED):
    """
    Returns ndarray (n_traj, 3*out_steps + 1) sorted best-first.
    Body-frame positions [x0..xN, y0..yN, z0..zN] + rel_cost.
    """
    rng  = np.random.RandomState()
    yaw_offsets   = np.linspace(-0.6, 0.6, n_traj)
    pitch_offsets = rng.uniform(-0.15, 0.15, n_traj)

    ts = np.arange(1, out_steps + 1) * speed * dt  # (steps,)

    dirs_bf = np.stack([
        np.cos(pitch_offsets) * np.cos(yaw_offsets),
        np.cos(pitch_offsets) * np.sin(yaw_offsets),
        np.sin(pitch_offsets)
    ], axis=1)                                      # (n_traj, 3)

    # Body-frame waypoints: (n_traj, steps, 3)
    pts_bf = dirs_bf[:, None, :] * ts[None, :, None]

    # World-frame waypoints for cost computation
    pts_wf = pos[None, None, :] + (pts_bf @ R_bw.T)   # (n_traj, steps, 3)

    costs = env.traj_costs(pts_wf)                 # (n_traj,)

    # Sort by cost
    order = np.argsort(costs)
    pts_bf = pts_bf[order]
    costs  = costs[order]

    # Normalise cost
    costs = costs / (costs[-1] + 1e-6)

    # Pack as [x0..xN, y0..yN, z0..zN, cost]
    trajs = np.concatenate([
        pts_bf[:, :, 0],    # (n_traj, steps)  x
        pts_bf[:, :, 1],    # (n_traj, steps)  y
        pts_bf[:, :, 2],    # (n_traj, steps)  z
        costs[:, None]      # (n_traj, 1)
    ], axis=1).astype(np.float32)
    return trajs


# ─── Single rollout ───────────────────────────────────────────────────────────
def simulate_rollout(env, rollout_dir, n_steps=200, seed=0,
                     speed=AVG_SPEED, noise_std=0.05):
    os.makedirs(os.path.join(rollout_dir, 'img'),          exist_ok=True)
    os.makedirs(os.path.join(rollout_dir, 'trajectories'), exist_ok=True)

    rng = np.random.RandomState(seed)

    # Reference trajectory (straight +x at 2 m height)
    ref_len = n_steps + 300
    ref_x   = np.linspace(0, (ref_len - 1) * speed * DT, ref_len)
    ref_pts = np.column_stack([ref_x,
                               np.zeros(ref_len),
                               np.full(ref_len, 2.0)])
    ref_df = pd.DataFrame(ref_pts, columns=['pos_x', 'pos_y', 'pos_z'])
    ref_df.to_csv(os.path.join(rollout_dir, 'reference_trajectory.csv'), index=False)

    # Drone state
    pos   = np.array([0.0, 0.0, 2.0])
    yaw   = 0.0
    odom_rows = []

    for step in range(n_steps):
        yaw += rng.normal(0, noise_std * 0.1)
        yaw  = np.clip(yaw, -0.5, 0.5)

        rot   = R_scipy.from_euler('zyx', [yaw, 0.0, 0.0])
        quat  = rot.as_quat()          # [qx, qy, qz, qw]
        R_bw  = rot.as_matrix()        # body→world (3×3)

        vel   = R_bw @ np.array([speed, 0.0, 0.0]) + rng.normal(0, noise_std, 3)
        omega = rng.normal(0, noise_std * 0.5, 3)

        odom_rows.append(list(pos) + list(quat) + list(vel) + list(omega))

        # ── Depth image ───────────────────────────────────────────────────
        depth_img = env.render_depth(pos, R_bw)
        cv2.imwrite(
            os.path.join(rollout_dir, 'img', 'depth_{:08d}.tif'.format(step+1)),
            depth_img)

        # ── RGB placeholder (all zeros; training uses use_rgb=False) ──────
        cv2.imwrite(
            os.path.join(rollout_dir, 'img', 'frame_left_{:08d}.png'.format(step+1)),
            np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8))

        # ── Trajectory labels ─────────────────────────────────────────────
        trajs = generate_trajectories_bf(env, pos, R_bw)
        np.save(
            os.path.join(rollout_dir, 'trajectories',
                         'trajectories_bf_{:08d}.npy'.format(step)),
            trajs)

        # Advance
        pos = pos + vel * DT
        pos[2] = max(0.5, pos[2] + rng.normal(0, 0.02))

    # ── Odometry CSV ──────────────────────────────────────────────────────
    cols = (['pos_x','pos_y','pos_z',
             'q_x','q_y','q_z','q_w',
             'vel_x','vel_y','vel_z',
             'omega_x','omega_y','omega_z'])
    pd.DataFrame(odom_rows, columns=cols).to_csv(
        os.path.join(rollout_dir, 'odometry.csv'), index=False)

    # ── Point cloud ───────────────────────────────────────────────────────
    pts = env.point_cloud()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(
        os.path.join(rollout_dir, 'pointcloud-unity.ply'), pcd)

    return n_steps


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir',      default='data')
    ap.add_argument('--n_train',      type=int, default=15)
    ap.add_argument('--n_val',        type=int, default=5)
    ap.add_argument('--n_steps',      type=int, default=200)
    ap.add_argument('--n_obstacles',  type=int, default=60)
    args = ap.parse_args()

    train_dir = os.path.join(args.out_dir, 'train')
    val_dir   = os.path.join(args.out_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)

    total = args.n_train + args.n_val
    print(f"\n=== Generating {args.n_train} train + {args.n_val} val rollouts "
          f"({args.n_steps} steps each) ===\n")

    for i in tqdm(range(total), desc='Rollouts'):
        seed = i * 17 + 3
        env  = Environment(n_obstacles=args.n_obstacles, seed=seed)
        if i < args.n_train:
            base, idx, split = train_dir, i, 'train'
        else:
            base, idx, split = val_dir, i - args.n_train, 'val'

        rdir = os.path.join(base, 'rollout_{:04d}'.format(idx))
        n    = simulate_rollout(env, rdir, n_steps=args.n_steps, seed=seed)
        tqdm.write(f'  [{split}] rollout_{idx:04d}: {n} steps')

    print(f'\nDone. Dataset: {args.out_dir}  (train: {train_dir}, val: {val_dir})')


if __name__ == '__main__':
    main()
