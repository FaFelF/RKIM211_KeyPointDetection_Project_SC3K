import numpy as np
import os
import json

BASE_PATH = "/home/felix/Hochschule/RKIM/KI in Prod/Projekt/Data Sets/KeypointNet_SC3K"
ANNOT_DIR = os.path.join(BASE_PATH, "annotations")
OUTPUT_DIR = "/home/felix/Hochschule/RKIM/KI in Prod/Projekt/Data Sets/KeypointNet_SC3K_sphere_poses"

# 24 Kameras auf der oberen Halbkugel (wie im ONet/ShapeNet-Standard):
# 3 Elevationen x 8 Azimute = 24 Blickwinkel, alle auf origin gerichtet.
# Alle Modelle bekommen exakt diese 24 Rotationen → gemeinsamer kanonischer Raum.
_ELEVATIONS_DEG = [15.0, 35.0, 60.0]   # obere Halbkugel, z > 0
_AZIMUTHS_DEG   = np.linspace(0, 360, 8, endpoint=False)  # 0, 45, 90 ... 315


def _look_at(eye):
    """Liefert die 3×3 Rotationsmatrix (world→camera) für eine Kamera
    an Position `eye`, die auf den Ursprung schaut.
    Konvention: kamera-Z zeigt zur Szene (OpenCV-Stil)."""
    world_up = np.array([0.0, 0.0, 1.0])
    z = eye / np.linalg.norm(eye)               # Blickrichtung weg vom Ursprung
    if abs(np.dot(z, world_up)) > 0.999:         # Kamera zeigt genau nach oben/unten
        world_up = np.array([0.0, 1.0, 0.0])
    x = np.cross(world_up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    return np.stack([x, y, z], axis=0)          # rows = Kamera-Achsen in Weltkoordinaten


def _build_sphere_cameras(r=1.5):
    """24 LookAt-Extrinsics [R|t] als (3×4)-Matrizen von der oberen Halbkugel."""
    cameras = []
    for elev_deg in _ELEVATIONS_DEG:
        for azim_deg in _AZIMUTHS_DEG:
            elev = np.radians(elev_deg)
            azim = np.radians(azim_deg)
            eye = r * np.array([
                np.cos(elev) * np.cos(azim),
                np.cos(elev) * np.sin(azim),
                np.sin(elev)
            ])
            R = _look_at(eye)
            t = -R @ eye                        # Kameraposition in Kamerakoordinaten
            mat = np.zeros((3, 4), dtype=np.float64)
            mat[:, :3] = R
            mat[:, 3]  = t
            cameras.append(mat)
    return cameras


CANONICAL_CAMERAS = _build_sphere_cameras()


def create_pose_file(model_id, category_id):
    target_dir = os.path.join(OUTPUT_DIR, category_id)
    os.makedirs(target_dir, exist_ok=True)

    pose_dict = {}
    intrinsic = np.eye(3, dtype=np.float64)

    for i, cam in enumerate(CANONICAL_CAMERAS):
        pose_dict[f'world_mat_{i}'] = cam
        pose_dict[f'camera_mat_{i}'] = intrinsic

    np.savez(os.path.join(target_dir, f"{model_id}.npz"), **pose_dict)


if __name__ == "__main__":
    # Sanity-Check: alle 24 Rotationen sollten det=1 haben
    for i, cam in enumerate(CANONICAL_CAMERAS):
        d = np.linalg.det(cam[:, :3])
        assert abs(d - 1.0) < 1e-6, f"view {i}: det={d}"
    print(f"24 Kugel-Kameras OK (det=1 für alle)")

    print(f"Generiere kanonische Sphere-Posen fuer alle Modelle ...")
    print(f"  Elevationen: {_ELEVATIONS_DEG} Grad")
    print(f"  Azimute:     {list(_AZIMUTHS_DEG.astype(int))} Grad")
    print(f"  Output: {OUTPUT_DIR}")

    total = 0
    for json_file in sorted(os.listdir(ANNOT_DIR)):
        if not json_file.endswith('.json'):
            continue
        with open(os.path.join(ANNOT_DIR, json_file), 'r') as f:
            data = json.load(f)
        for entry in data:
            create_pose_file(entry['model_id'], entry['class_id'])
            total += 1

    print(f"Fertig: {total} Pose-Dateien erstellt.")
