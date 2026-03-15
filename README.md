# MeshGPT — Point Cloud Reconstruction & Mesh Generation

> A transformer-based autoencoder that learns to reconstruct 3D point clouds from a compact latent representation, then converts those reconstructions into watertight triangular meshes using alpha-shape surface recovery.

---

## Overview

MeshGPT takes raw 3D mesh models (ModelNet10), converts them into point clouds, and trains a **TransformerFoldingAE** — a hybrid architecture combining a Point Transformer encoder with a two-stage FoldingNet++ decoder — to reconstruct those point clouds from a 256-dimensional bottleneck. Reconstructed point clouds are then lifted back into triangular meshes via alpha shapes, producing export-ready `.ply` and `.obj` files.

The project demonstrates a full 3D understanding pipeline: **mesh → point cloud → latent code → reconstructed point cloud → mesh**.

---

## Table of Contents

- [Motivation](#motivation)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Example Outputs](#example-outputs)
- [Contributing](#contributing)
- [License](#license)

---

## Motivation

Generative models for 3D shapes face a fundamental challenge: meshes are irregular, unordered, and vary in topology — making them hard to model directly. Point clouds offer a permutation-invariant alternative, but converting learned point cloud representations back into usable meshes is non-trivial.

This project addresses that gap by:

1. Using a **transformer encoder** to capture global shape structure via self-attention over unordered point sets.
2. Using a **FoldingNet++ decoder** to unfold a learned 2D grid into 3D space in two stages, with residual refinement.
3. Applying **alpha shape reconstruction** to recover a triangulated surface from the reconstructed point cloud.

---

## Features

- **TransformerFoldingAE** — end-to-end autoencoder for 3D point clouds
- **Point Transformer Encoder** — 4-layer multi-head self-attention with adaptive max pooling
- **FoldingNet++ Decoder** — two-stage folding with learned 2D offsets and residual refinement
- **Multi-component loss** — Chamfer distance + repulsion + Laplacian smoothness
- **Alpha shape mesh export** — reconstructions exported as `.ply`, `.obj`, and `.mtl`
- **ICP-aligned evaluation** — precision, recall, and F1 metrics with optional rigid alignment
- **Apple Silicon support** — MPS device detection alongside CUDA and CPU
- **ModelNet10 preprocessing** — mesh → normalized point cloud pipeline included

---

## Architecture

```
Input Point Cloud (B, 2048, 3)
         │
         ▼
┌─────────────────────────────┐
│   PointTransformerEncoder   │
│                             │
│  Linear: 3 → 128            │
│  4× TransformerEncoderLayer │
│    - Multi-Head Attention   │
│      (4 heads, d=128)       │
│    - FFN: 128 → 256 → 128   │
│    - LayerNorm + Residual   │
│  AdaptiveMaxPool → Flatten  │
│  Linear: 128 → 256          │
└────────────┬────────────────┘
             │  Latent Vector (B, 256)
             ▼
┌─────────────────────────────┐
│   FoldingNetPPDecoder       │
│                             │
│  Base Grid: 45×45 (2D)      │
│  Learned offsets via MLP    │
│                             │
│  Stage 1 Folding:           │
│    [grid | latent] → MLP    │
│    512 → 512 → 3D points    │
│                             │
│  Stage 2 Folding:           │
│    [fold1 | latent] → MLP   │
│    512 → 512 → refined 3D   │
│                             │
│  Residual Refinement:       │
│    3 → 64 → 3 (scale 0.1)   │
└────────────┬────────────────┘
             │
             ▼
Reconstructed Point Cloud (B, 2048, 3)
             │
             ▼
  Alpha Shape Reconstruction
             │
             ▼
   Triangular Mesh (.ply / .obj)
```

**Loss Function:**

```
L = L_chamfer + 0.15 × L_repulsion + 0.05 × L_smoothness
```

| Component | Description |
|-----------|-------------|
| `L_chamfer` | Bidirectional minimum point-to-point distances |
| `L_repulsion` | k-NN exponential decay; prevents point collapse |
| `L_smoothness` | Laplacian term; penalizes deviation from neighbor mean |

---

## Project Structure

```
MeshGPT/
├── scripts/
│   ├── transformer_folding_ae.py   # Model architecture + loss functions
│   ├── pytorch_ds.py               # ModelNet10PC dataset class (auto train/val/test split)
│   └── preprocessing.py            # OFF mesh → normalized .npy point cloud
│
├── training_foldingnet_ae.py       # Training loop (Adam, 80 epochs, batch=4)
├── eval_transfold.py               # Evaluation: P/R/F1 with optional ICP alignment
├── vis_test_foldingnet_ae.py       # Interactive Open3D visualization of reconstructions
├── point_cloud_save.py             # Export reconstructed point clouds to .npy
├── tri_meshes_alpha.py             # Convert .npy point clouds → alpha shape meshes
├── testing_pointcloud.py           # Quick viewer for a raw point cloud
├── testing_triangular_mesh.py      # Quick viewer for an exported mesh
├── metrics.py                      # precision_recall_f1 implementation
│
├── data/
│   └── modelnet10_pc_2048/         # Preprocessed point clouds (.npy, 2048 pts each)
│       └── <class>/train|test/
│
├── checkpoints_foldingnet/         # Per-epoch model checkpoints (.pth)
├── checkpoints_transformer/        # Transformer-specific checkpoints
├── exported_pointclouds/           # Saved original + reconstructed point clouds
├── exported_meshes_alpha/          # Final mesh outputs (.ply, .obj, .mtl)
│
├── requirements.txt
└── log.txt                         # Training run log
```

---

## Installation

**Requirements:** Python 3.8+, pip

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/MeshGPT.git
cd MeshGPT

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3):** MPS acceleration is detected automatically. No extra steps needed.
>
> **CUDA:** Ensure your PyTorch installation matches your CUDA version. Visit [pytorch.org](https://pytorch.org) for install options.

### Dataset Setup

Download [ModelNet10](http://modelnet.cs.princeton.edu/) and preprocess it into point clouds:

```bash
python scripts/preprocessing.py \
  --input_dir data/ModelNet10 \
  --output_dir data/modelnet10_pc_2048 \
  --points 2048
```

This samples 2048 points per mesh surface and saves normalized `.npy` files maintaining the class/train/test directory structure.

---

## Quick Start

```bash
# Step 1 — Preprocess raw meshes (skip if data/modelnet10_pc_2048 exists)
python scripts/preprocessing.py --input_dir data/ModelNet10 --output_dir data/modelnet10_pc_2048

# Step 2 — Train the autoencoder
python training_foldingnet_ae.py

# Step 3 — Visualize a reconstruction (interactive Open3D window)
python vis_test_foldingnet_ae.py

# Step 4 — Export reconstructed point clouds
python point_cloud_save.py

# Step 5 — Convert point clouds to triangular meshes
python tri_meshes_alpha.py
```

Meshes are saved to `exported_meshes_alpha/<class>/` in `.ply`, `.obj`, and `.mtl` formats.

---

## Usage

### Training

```bash
python training_foldingnet_ae.py
```

Key hyperparameters (edit at the top of the file):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_points` | 2048 | Points per cloud |
| `d_model` | 128 | Transformer embedding dimension |
| `num_layers` | 4 | Transformer encoder depth |
| `nhead` | 4 | Attention heads |
| `latent_dim` | 256 | Bottleneck size |
| `num_epochs` | 80 | Training epochs |
| `batch_size` | 4 | Batch size |
| `lr` | 1e-4 | Adam learning rate |

Checkpoints are saved after every epoch to `checkpoints_foldingnet/foldingnet_epoch{N}.pth`.

---

### Evaluation

```bash
python eval_transfold.py
```

Computes **Precision**, **Recall**, and **F1** over the test set at a distance threshold of `tau=0.03`. Each sample is evaluated both raw and after ICP alignment to account for rigid-body differences.

```
Sample 42 | Chair
  Raw  -> P: 0.5234 | R: 0.4892 | F1: 0.5060
  ICP  -> P: 0.6120 | R: 0.5847 | F1: 0.5983

Average over test set:
  Raw  -> P: 0.5102 | R: 0.4744 | F1: 0.4916
  ICP  -> P: 0.6034 | R: 0.5711 | F1: 0.5868
```

---

### Visualization

```bash
python vis_test_foldingnet_ae.py
```

Opens an interactive Open3D window showing:
- **Blue** — original input point cloud
- **Red** — ICP-aligned reconstruction

Metrics are printed to the console alongside the visualization.

---

### Mesh Export

```bash
# Export reconstructed point clouds
python point_cloud_save.py

# Convert point clouds to alpha shape meshes
python tri_meshes_alpha.py
```

Alpha shape parameters (edit in `tri_meshes_alpha.py`):

| Parameter | Default | Effect |
|-----------|---------|--------|
| `alpha` | 0.09 | Smaller = more detail; larger = smoother surface |
| `voxel_size` | 0.02 | Downsampling voxel size (applied if >5000 points) |

Output per sample:
```
exported_meshes_alpha/
  chair/
    sample_001_recon.ply
    sample_001_recon.obj
    sample_001_recon.mtl
```

---

### Quick Viewers

```bash
# View a raw point cloud
python testing_pointcloud.py

# View an exported mesh with wireframe overlay
python testing_triangular_mesh.py
```

---

## Evaluation

Reconstruction quality is measured using point-level **Precision**, **Recall**, and **F1** at a configurable distance threshold `tau`:

- **Precision** — fraction of predicted points within `tau` of any ground-truth point
- **Recall** — fraction of ground-truth points within `tau` of any predicted point
- **F1** — harmonic mean of precision and recall

ICP (Iterative Closest Point) alignment is applied before the second round of metrics to isolate shape quality from pose differences.

```python
from metrics import precision_recall_f1

p, r, f1 = precision_recall_f1(pred_pc, gt_pc, tau=0.03)
```

---

## Example Outputs

The pipeline produces triangulated meshes from learned latent representations of ModelNet10 shapes. Output meshes use:
- **Vertex color:** soft blue-grey `[0.7, 0.75, 0.8]`
- **Transparency:** 0.85 (encoded in `.mtl`)
- **Wireframe overlay:** enabled in the viewer

Exported files per sample: `.ply` (binary), `.obj` (Wavefront), `.mtl` (material).

---

## Contributing

Contributions are welcome. To get started:

1. Fork the repository and create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, keeping code style consistent with the existing scripts.

3. Test your changes end-to-end (preprocessing → training → eval).

4. Open a pull request with a clear description of what was changed and why.

**Areas open for contribution:**
- Additional decoder architectures (e.g., flow-based, diffusion)
- Support for ShapeNet or other datasets
- Configurable training via CLI args or a config file
- Marching cubes as an alternative surface reconstruction method
- Quantitative benchmarks against other point cloud autoencoders

---

## License

This project is released under the [MIT License](LICENSE).

---

*Built on [ModelNet10](http://modelnet.cs.princeton.edu/) · PyTorch · Open3D · trimesh*
