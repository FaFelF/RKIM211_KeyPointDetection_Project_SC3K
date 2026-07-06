import numpy as np
import os
import json
from scipy.spatial.transform import Rotation as R

BASE_PATH = "/path/to/KeypointNet_SC3K"
ANNOT_DIR = os.path.join(BASE_PATH, "annotations")
OUTPUT_DIR = os.path.join(BASE_PATH, "poses")

def create_pose_file(model_id, category_id, seed_base=None):
    target_dir = os.path.join(OUTPUT_DIR, category_id)
    os.makedirs(target_dir, exist_ok=True)
    
    # Deterministischer Seed pro Modell – so sind die Posen reproduzierbar
    if seed_base is None:
        seed_base = hash(model_id) % (2**32)
    rng = np.random.RandomState(seed_base)
    
    pose_dict = {}
    intrinsic = np.eye(3, dtype=np.float64)
    
    for i in range(24):
        # Zufällige 3D-Rotation
        random_rot = R.random(random_state=rng).as_matrix().astype(np.float64)
        # Translation lassen wir auf 0 (Objekte sind bei KeypointNet zentriert)
        extrinsic_3x4 = np.zeros((3, 4), dtype=np.float64)
        extrinsic_3x4[:3, :3] = random_rot
        
        pose_dict[f'world_mat_{i}'] = extrinsic_3x4
        pose_dict[f'camera_mat_{i}'] = intrinsic
    
    save_path = os.path.join(target_dir, f"{model_id}.npz")
    np.savez(save_path, **pose_dict)

if __name__ == "__main__":
    print("Generiere zufällige Posen pro Modell...")
    for json_file in os.listdir(ANNOT_DIR):
        if json_file.endswith('.json'):
            with open(os.path.join(ANNOT_DIR, json_file), 'r') as f:
                data = json.load(f)
            for entry in data:
                create_pose_file(entry['model_id'], entry['class_id'])
    print("✅ Alle Poses neu generiert mit zufälligen Rotationen.")