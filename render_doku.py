"""Rendert PLY-Visualisierungen als hochwertige 2D-Projektion (kein OpenGL noetig)."""
import os
import sys
import glob
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection

VERTS_PER_SPHERE = 762
KP_RADIUS_THRESH = 0.02


def _look_at_matrix(eye, target, up=np.array([0, 0, 1.0])):
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=0)


def _extract_spheres(ply_path):
    m = trimesh.load(ply_path)
    verts = np.array(m.vertices)
    colors = np.array(m.visual.vertex_colors)[:, :3] / 255.0
    n = len(verts) // VERTS_PER_SPHERE
    centers = np.zeros((n, 3))
    cols = np.zeros((n, 3))
    radii = np.zeros(n)
    for i in range(n):
        s, e = i * VERTS_PER_SPHERE, (i + 1) * VERTS_PER_SPHERE
        chunk = verts[s:e]
        c = chunk.mean(axis=0)
        centers[i] = c
        cols[i] = colors[s]
        radii[i] = np.max(np.linalg.norm(chunk - c, axis=1))
    is_kp = radii > KP_RADIUS_THRESH
    return centers, cols, is_kp


def render(ply_path, output_path, elev=25, azim=-60):
    centers, cols, is_kp = _extract_spheres(ply_path)

    centroid = centers.mean(axis=0)
    span = np.max(np.ptp(centers, axis=0))
    dist = span * 2.5

    elev_r, azim_r = np.radians(elev), np.radians(azim)
    eye = centroid + dist * np.array([
        np.cos(elev_r) * np.cos(azim_r),
        np.cos(elev_r) * np.sin(azim_r),
        np.sin(elev_r),
    ])
    R = _look_at_matrix(eye, centroid)
    cam = (centers - centroid) @ R.T
    depth = cam[:, 2]

    pc_mask = ~is_kp
    kp_mask = is_kp

    fig, ax = plt.subplots(figsize=(14, 10), dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Point cloud: sort far-to-near, depth-based size and alpha
    pc_idx = np.where(pc_mask)[0]
    order = np.argsort(-depth[pc_idx])
    pc_sorted = pc_idx[order]

    d = depth[pc_sorted]
    d_min, d_max = d.min(), d.max()
    d_norm = (d - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(d)
    nearness = 1.0 - d_norm

    sizes = 5 + 12 * nearness
    alphas = 0.2 + 0.6 * nearness
    pc_colors = np.column_stack([cols[pc_sorted], alphas])

    ax.scatter(cam[pc_sorted, 0], cam[pc_sorted, 1],
               c=pc_colors, s=sizes, linewidths=0, zorder=1)

    # Keypoints: sort far-to-near, draw on top
    kp_idx = np.where(kp_mask)[0]
    kp_order = kp_idx[np.argsort(-depth[kp_idx])]
    ax.scatter(cam[kp_order, 0], cam[kp_order, 1],
               c=cols[kp_order], s=400, alpha=1.0,
               edgecolors='black', linewidths=1.2, zorder=5)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.05,
                dpi=150, facecolor='white')
    plt.close()


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/2026-06-23_09-44-40_airplane_baseline_test"
    elev = float(sys.argv[2]) if len(sys.argv) > 2 else 25
    azim = float(sys.argv[3]) if len(sys.argv) > 3 else -60
    ply_dir = os.path.join(run_dir, "generic_visualizations", "ply")
    out_dir = os.path.join(run_dir, "generic_visualizations", "png_doku")
    os.makedirs(out_dir, exist_ok=True)

    plys = sorted(glob.glob(os.path.join(ply_dir, "*.ply")))
    print("Rendere {} PLY-Dateien (elev={}, azim={}) -> {}".format(
        len(plys), elev, azim, out_dir))

    for ply in plys:
        name = os.path.splitext(os.path.basename(ply))[0] + ".png"
        out = os.path.join(out_dir, name)
        render(ply, out, elev=elev, azim=azim)
        print("  " + name)

    print("Fertig: {} Bilder".format(len(plys)))
