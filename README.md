# SC3K – Self-supervised and Coherent 3D Keypoints Estimation

This repository is based on the paper [SC3K: Self-supervised and Coherent 3D Keypoints Estimation from Rotated, Noisy, and Decimated Point Cloud Data](https://openaccess.thecvf.com/content/ICCV2023/papers/Zohaib_SC3K_Self-supervised_and_Coherent_3D_Keypoints_Estimation_from_Rotated_Noisy_ICCV_2023_paper.pdf) (ICCV 2023) and was adapted and evaluated as part of a university project.

<p align="center">
  <img src="images/over_view.gif" alt="over_view" width="600"/>
</p>

---

## What was done

The original SC3K code was set up, trained and evaluated on the **airplane** class of the KeypointNet dataset. The main focus was on the generation of camera poses, which the paper describes but does not provide. Three different pose generation strategies were implemented and compared:

### Pose Generation Approaches

| # | Approach | Description |
|---|---|---|
| 1 | **Random Poses** | 24 uniformly random rotation matrices per model (reproduced from literature) |
| 2 | **Fixed Poses** | 24 identical rotations for all models (no model-specific randomness) |
| 3 | **Sphere Poses** | 24 LookAt cameras placed on the upper hemisphere (3 elevations × 8 azimuths), as described in the paper |

The sphere pose approach (`generate_canonical_poses.py`) follows the camera convention described in the SC3K paper: cameras are placed at elevations of 15°, 35° and 60° with 8 evenly spaced azimuths each, all looking toward the origin.

---

## Results (Airplane class)

All models were trained for approximately 91–102 epochs (~10 hours each on a laptop GPU).

| Approach | Epochs | DAS ↑ | Pose Error ↓ | Coverage ↑ | Inclusivity ↑ |
|---|---|---|---|---|---|
| Random Poses | 78 | 19.80% | 133.3° | 93.19% | 90.63% |
| Fixed Poses | 102 | 43.53% | 140.7° | 97.19% | 91.80% |
| **Sphere Poses** | **91** | **61.34%** | **1.02°** | **98.12%** | **86.50%** |

The paper reports ~70% DAS. The remaining gap is likely due to disabled augmentations (`gaussian_noise`, `down_sample`) and the fact that the exact pose angles used in the paper are not published.

---

## Installation

Requires Python 3.6+, PyTorch 1.10.1, and a CUDA-capable GPU.

```bash
git clone <this-repo>
cd SC3K

conda env create -f sc3k_environment.yml
conda activate sc3k
```

---

## Dataset Setup

Download the [KeypointNet dataset](https://github.com/qq456cvb/KeypointNet) and place it in the following structure:

```
dataset/
  annotations/
    airplane.json
  pcds/
    02691156/
      *.pcd
  poses/         ← generate with one of the scripts below
    02691156/
      *.npz
  splits/
    train.txt
    val.txt
    test.txt
```

A small sample set (13 models) is included in `dataset/` for testing the pipeline.

---

## Pose Generation

Two pose generation scripts are provided:

**Random Poses** (24 random rotations per model, reproducible via seed):
```bash
python generate_sc3k_poses.py
```

**Sphere Poses** (24 LookAt cameras on the upper hemisphere, same for all models):
```bash
python generate_canonical_poses.py
```

Set the output path in the script before running. Then update `poses_root` in `config/config.yaml` accordingly.

---

## Training

Edit `config/config.yaml`:

```yaml
split: train
class_name: airplane       # class to train
batch_size: 8
max_epoch: 100
poses_root: /path/to/poses
```

Then run:
```bash
python train.py
```

Training progress and the best model weights are saved under `runs/`.

---

## Evaluation

Edit `config/config.yaml`:

```yaml
split: test
save_results: True
data:
  best_model_path: runs/<run_name>/Best_airplane_10kp.pth
  poses_root: /path/to/poses
```

Then run:
```bash
python test.py
```

Metrics (DAS, Coverage, Inclusivity, Pose Error) are logged to `runs/<run_name>/test.log`.  
Visualizations (PLY + PNG) are saved under `runs/<run_name>/generic_visualizations/`.

---

## Visualization

To render existing PLY visualizations from a custom camera angle (no display required):

```bash
python render_doku.py <run_dir> [elev] [azim]
# Example:
python render_doku.py runs/2026-06-23_09-44-40_airplane_baseline_test 25 -60
```

---

## Pretrained Models

Trained model weights for the airplane class are provided in `models/airplane/`:

```
models/
  airplane/
    random_poses/   → Best_airplane_10kp.pth   (78 ep,  DAS 19.80%)
    fix_poses/      → Best_airplane_10kp.pth   (102 ep, DAS 43.53%)
    sphere_poses/   → Best_airplane_10kp.pth   (91 ep,  DAS 61.34%)
```

To use a model, set `best_model_path` in `config/config.yaml` to the desired `.pth` file.

---

## Original Paper

```
@inproceedings{zohaib2023sc3k,
  title={SC3K: Self-supervised and Coherent 3D Keypoints Estimation from Rotated, Noisy, and Decimated Point Cloud Data},
  author={Zohaib, Mohammad and Del Bue, Alessio},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={22509--22519},
  year={2023}
}
```
