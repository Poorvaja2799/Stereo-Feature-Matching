Stereo Feature Matching
=======================

Overview
--------
- Computational toolkit for stereo feature matching and basic 3D vision.
- Implements Harris corners, SIFT-like descriptors, feature matching with ratio test, camera projection matrix, fundamental matrix (normalized 8-point), and RANSAC-based homography.
- Includes a notebook demonstrating the pipeline and sample datasets for experimentation.

Repository Structure
--------------------
- project-2.ipynb — end-to-end walkthrough and experiments
- conda/environment.yml — reproducible environment
- conda/install.sh — one-shot installer using mamba and editable install
- src/vision/
	- part1_harris_corner.py — Harris corners and NMS
	- part2_feature_matching.py — distance matrix + Lowe ratio matching
	- part3_sift_descriptor.py — 128-D SIFT-like descriptors
	- part4_projection_matrix.py — DLT projection matrix + camera center
	- part5_fundamental_matrix.py — normalized 8-point F estimation
	- part6_ransac.py — RANSAC homography wrapper and iteration calculator
	- utils.py — image I/O, visualization, helpers
- data/ — sample correspondences and images for evaluation
- results/ — place for outputs and figures

Quick Start
-----------
1) Create the environment

```bash
# macOS/Linux
bash conda/install.sh
```

This will:
- Ensure `mamba` is available.
- Create the `cv_proj2` environment from `conda/environment.yml`.
- Install this repo in editable mode.

2) Manual setup (alternative)

```bash
conda env create -f conda/environment.yml
conda activate cv_proj2
python -m pip install -e .
```

3) Launch the demo notebook

```bash
conda activate cv_proj2
jupyter notebook project-2.ipynb
```

Data
----
The `data/` folder includes a few small datasets and point lists for evaluating matching and geometry:
- `Episcopal_Gaudi/`, `Notre_Dame/`, `Woodruff_Dorm/` — classic feature-matching pairs with provided 2D correspondences.
- `argoverse_*` and `vo_seq_argoverse_*` — sample frames for visual odometry style matching.