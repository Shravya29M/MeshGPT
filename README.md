# MeshGPT — Point Cloud Reconstruction & Mesh Generation

> We train a transformer autoencoder to learn compressed representations of 3D shapes, then reconstruct them as triangular meshes — going all the way from raw ModelNet10 files to exported `.ply` and `.obj` geometry.

---

## What this is

Working with 3D shapes is messy. Raw meshes are irregular and vary wildly in topology, which makes them hard to feed into learning pipelines. So we convert them into point clouds first — simpler, uniform, and easier to reason about — train a model to compress and reconstruct them, and then convert the output back into actual meshes.

The full pipeline looks like this:

**raw mesh → point cloud → 256-dim latent code → reconstructed point cloud → triangular mesh**

The model is a **TransformerFoldingAE**: a Point Transformer encoder that reads an unordered set of 2048 points and squeezes them into a latent vector, paired with a FoldingNet++ decoder that unfolds a flat 2D grid into 3D space in two passes. The final meshes come from running alpha shape reconstruction on the decoded output.

---

## Table of Contents

- [Why we built it this way](#why-we-built-it-this-way)
- [What's included](#whats-included)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Running the pipeline](#running-the-pipeline)
- [Evaluating results](#evaluating-results)
- [Example outputs](#example-outputs)
- [Contributing](#contributing)
- [License](#license)

---

## Why we built it this way

The core challenge with 3D shape generation: meshes don't fit neatly into standard deep learning frameworks. They're unstructured, have variable vertex counts, and topology can differ dramatically between shapes.

We sidestep that by working in point cloud space, where everything is just a fixed-size matrix of XYZ coordinates. The tricky part is getting *back* to a mesh — that's where alpha shapes come in. Instead of trying to predict mesh connectivity directly, we let the reconstructed point cloud define the surface implicitly and recover the triangulation geometrically.

The three-part loss function reflects this too: Chamfer distance makes sure points land in the right place, repulsion keeps them from collapsing into clusters, and smoothness nudges neighboring points to behave coherently.

---

## What's included

- **TransformerFoldingAE** — the full autoencoder, encoder + decoder
- **Point Transformer Encoder** — 4 layers of multi-head self-attention with adaptive max pooling
- **FoldingNet++ Decoder** — two-stage folding from a learned 2D grid, with a small residual refinement on top
- **Multi-term loss** — Chamfer + repulsion + Laplacian smoothness, weighted and tunable
- **Mesh export** — alpha shape reconstruction to `.ply`, `.obj`, and `.mtl`
- **ICP-aligned evaluation** — precision/recall/F1 with and without rigid alignment
- **Apple Silicon / CUDA / CPU** — device detection is automatic, nothing to configure
- **Preprocessing script** — converts raw ModelNet10 `.off` files into normalized `.npy` point clouds

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

**Loss:**

```
L = L_chamfer  +  0.15 × L_repulsion  +  0.05 × L_smoothness
```

| Term | What it does |
|------|-------------|
| `L_chamfer` | Pulls predicted points toward ground-truth and vice versa |
| `L_repulsion` | Pushes points apart so they don't pile up in one spot |
| `L_smoothness` | Encourages each point to stay near the mean of its neighbors |

---

## Project structure

```
MeshGPT/
├── scripts/
│   ├── transformer_folding_ae.py   # Model definition + all three loss functions
│   ├── pytorch_ds.py               # Dataset class — handles splitting and normalization
│   └── preprocessing.py            # Converts ModelNet10 .off files to .npy point clouds
│
├── training_foldingnet_ae.py       # Training loop
├── eval_transfold.py               # Runs evaluation over the test set
├── vis_test_foldingnet_ae.py       # Opens an Open3D window with original vs reconstruction
├── point_cloud_save.py             # Saves reconstructed point clouds to disk
├── tri_meshes_alpha.py             # Runs alpha shape reconstruction on saved point clouds
├── testing_pointcloud.py           # Quick sanity check — loads and displays one point cloud
├── testing_triangular_mesh.py      # Quick sanity check — loads and displays one mesh
├── metrics.py                      # precision_recall_f1 function
│
├── data/
│   └── modelnet10_pc_2048/         # Where preprocessed point clouds live
│       └── <class>/train|test/*.npy
│
├── checkpoints_foldingnet/         # Saved checkpoints, one per epoch
├── checkpoints_transformer/
├── exported_pointclouds/           # Raw .npy exports (original + reconstructed pairs)
├── exported_meshes_alpha/          # Final mesh outputs
│
├── requirements.txt
└── log.txt
```

---

## Getting started

**Python 3.8+ required.**

```bash
# Clone
git clone https://github.com/yourusername/MeshGPT.git
cd MeshGPT

# Set up a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Mac (M1/M2/M3):** MPS is detected and used automatically.
> **CUDA:** Make sure your PyTorch version matches your CUDA version — see [pytorch.org](https://pytorch.org) for the right install command.

### Dataset

Download [ModelNet10](http://modelnet.cs.princeton.edu/) and run the preprocessing script to convert the raw `.off` meshes into point clouds:

```bash
python scripts/preprocessing.py \
  --input_dir data/ModelNet10 \
  --output_dir data/modelnet10_pc_2048 \
  --points 2048
```

This samples 2048 surface points per mesh, normalizes each cloud to a unit sphere, and saves the result as `.npy` files in the same class/train/test structure as the original dataset.

---

## Running the pipeline

Once your data is set up, the whole pipeline runs in five steps:

```bash
# 1. Train
python training_foldingnet_ae.py

# 2. Spot-check a reconstruction visually
python vis_test_foldingnet_ae.py

# 3. Evaluate on the test set
python eval_transfold.py

# 4. Export reconstructed point clouds
python point_cloud_save.py

# 5. Convert them to meshes
python tri_meshes_alpha.py
```

Meshes land in `exported_meshes_alpha/<class>/` as `.ply`, `.obj`, and `.mtl` files.

---

### Training details

```bash
python training_foldingnet_ae.py
```

Hyperparameters are set at the top of the file:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_points` | 2048 | Points per cloud |
| `d_model` | 128 | Transformer embedding size |
| `num_layers` | 4 | Encoder depth |
| `nhead` | 4 | Attention heads |
| `latent_dim` | 256 | Bottleneck size |
| `num_epochs` | 80 | |
| `batch_size` | 4 | |
| `lr` | 1e-4 | Adam |

A checkpoint is saved after every epoch to `checkpoints_foldingnet/foldingnet_epoch{N}.pth`, so you can resume or roll back to any point in training.

---

### Visualization

```bash
python vis_test_foldingnet_ae.py
```

Picks a random test sample, runs it through the model, and opens an Open3D window with the original (blue) and the ICP-aligned reconstruction (red) side by side. Precision/recall/F1 are printed to the console.

---

### Mesh export

```bash
python point_cloud_save.py   # saves original + reconstructed .npy pairs
python tri_meshes_alpha.py   # runs alpha shapes on the reconstructions
```

You can tune the surface reconstruction in `tri_meshes_alpha.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `alpha` | 0.09 | Lower = more detail, higher = smoother (more holes) |
| `voxel_size` | 0.02 | Downsampling applied if the cloud has >5000 points |

---

### Quick viewers

```bash
python testing_pointcloud.py      # view a raw point cloud
python testing_triangular_mesh.py # view a mesh with wireframe overlay
```

---

## Evaluating results

We use point-level **Precision**, **Recall**, and **F1** at a distance threshold `tau=0.03` (roughly 3% of the unit sphere radius):

- **Precision** — what fraction of your predicted points are actually close to the ground truth?
- **Recall** — what fraction of the ground truth points did you actually cover?
- **F1** — the harmonic mean, balancing both

We run metrics twice: once raw, and once after ICP alignment. ICP removes any rigid-body offset between the reconstruction and the original, so the second number isolates shape quality from pose.

```python
from metrics import precision_recall_f1

p, r, f1 = precision_recall_f1(pred_pc, gt_pc, tau=0.03)
```

Example output from `eval_transfold.py`:

```
Sample 42 | Chair
  Raw  -> P: 0.5234 | R: 0.4892 | F1: 0.5060
  ICP  -> P: 0.6120 | R: 0.5847 | F1: 0.5983
```

---

## Example outputs

The exported meshes are triangulated surfaces reconstructed from the decoder's output. Each one is saved with a soft blue-grey vertex color and slight transparency (α=0.85), and the viewer shows a wireframe overlay so you can see the triangle structure.

Output per sample:
```
exported_meshes_alpha/
  chair/
    sample_001_recon.ply
    sample_001_recon.obj
    sample_001_recon.mtl
```

---

## Contributing

If you want to extend this, here are some directions worth exploring:

- **Alternative decoders** — flow-based or diffusion-based decoders instead of FoldingNet
- **More datasets** — ShapeNet55 would be a natural next step
- **CLI config** — right now hyperparameters live in the training script; a config file or argparse would make experiments cleaner
- **Surface reconstruction** — marching cubes or Poisson reconstruction as alternatives to alpha shapes
- **Benchmarks** — comparison against FoldingNet, PCN, or other point cloud autoencoders

To contribute:

```bash
git checkout -b feature/your-feature-name
```

Make your changes, test end-to-end (preprocessing → training → eval), and open a pull request explaining what you changed and why.

---

## License

[MIT](LICENSE)

---

*Built with PyTorch · Open3D · trimesh · ModelNet10*
