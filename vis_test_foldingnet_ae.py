#!/usr/bin/env python3
import torch
import numpy as np
import open3d as o3d
import os

from scripts.pytorch_ds import ModelNet10PC
from scripts.transformer_folding_ae import TransformerFoldingAE
from metrics import precision_recall_f1


# -----------------------------
# DEVICE
# -----------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------
# Convert numpy array → Open3D point cloud
# -----------------------------
def to_o3d(pc, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.paint_uniform_color(color)
    return pcd


# -----------------------------
# ICP alignment
# -----------------------------
def icp_align(src_np, tgt_np, threshold=0.05):
    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()

    src.points = o3d.utility.Vector3dVector(src_np)
    tgt.points = o3d.utility.Vector3dVector(tgt_np)

    reg = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    src.transform(reg.transformation)
    return np.asarray(src.points)


# -----------------------------
# SAFE CHECKPOINT LOADING
# Works for:
#  - PyTorch >= 2.6 (weights_only default)
#  - old style state_dict files
#  - dict checkpoints from Taichi training
# -----------------------------
def load_checkpoint_safely(model, path, device):
    print(f"\nLoading checkpoint: {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # older PyTorch fallback
        checkpoint = torch.load(path, map_location=device)

    # If the checkpoint is dict-like and includes model_state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("✓ Loaded: dict checkpoint with model_state_dict")
        state_dict = checkpoint["model_state_dict"]
    else:
        print("✓ Loaded: raw state_dict")
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model


# -----------------------------
# MAIN VISUALIZER + EXPORT
# -----------------------------
def main():
    device = get_device()
    print("Using device:", device)

    # -----------------------------
    # FIXED INDEX FOR CONSISTENT COMPARISON
    # -----------------------------
    fixed_index = 90

    # -----------------------------
    # Load dataset
    # -----------------------------
    dataset = ModelNet10PC("data/modelnet10_pc_2048", split="test")
    print("Total test samples:", len(dataset))
    print("Using index:", fixed_index)

    original = dataset[fixed_index]  # numpy (2048, 3)

    # -----------------------------
    # Load model + checkpoint
    # -----------------------------

    # ckpt_path = "checkpoints_foldingnet/foldingnet_epoch280.pth"  # you can change this path
    ckpt_path = "checkpoints_taichi_pc/best_model.pth"


    model = TransformerFoldingAE(num_points=2048, latent_dim=256).to(device)
    model = load_checkpoint_safely(model, ckpt_path, device)
    model.eval()

    # -----------------------------
    # Forward pass
    # -----------------------------
    with torch.no_grad():
        inp = torch.tensor(original).unsqueeze(0).to(device)
        recon, _ = model(inp)
        recon = recon.squeeze(0).cpu().numpy()

    # metrics before ICP
    orig_t = torch.tensor(original).float()
    recon_t = torch.tensor(recon).float()
    p0, r0, f0 = precision_recall_f1(recon_t, orig_t)

    print("\n=== Raw (Unaligned) Metrics ===")
    print("Precision:", float(p0))
    print("Recall:", float(r0))
    print("F1:", float(f0))

    # ICP alignment
    recon_aligned = icp_align(recon, original)
    recon_align_t = torch.tensor(recon_aligned).float()
    p1, r1, f1 = precision_recall_f1(recon_align_t, orig_t)

    print("\n=== After ICP Alignment ===")
    print("Precision:", float(p1))
    print("Recall:", float(r1))
    print("F1:", float(f1))

    # -----------------------------
    # EXPORT POINT CLOUDS FOR MESHING
    # -----------------------------
    # export_dir = "exported_pointclouds/foldingnet_ae_base"
    export_dir = "exported_pointclouds/improved_phyics_pc"
    os.makedirs(export_dir, exist_ok=True)

    np.save(os.path.join(export_dir, f"sample_{fixed_index}_orig.npy"), original)
    np.save(os.path.join(export_dir, f"sample_{fixed_index}_recon.npy"), recon_aligned)

    print(f"\n✓ Exported point clouds to: {export_dir}")

    # -----------------------------
    # VISUALIZE USING OPEN3D GUI
    # -----------------------------
    from open3d.visualization import gui

    app = gui.Application.instance
    app.initialize()

    win = o3d.visualization.O3DVisualizer(
        "FoldingNetAE: Original + Reconstruction (Aligned)",
        1280, 720
    )

    orig_pcd = to_o3d(original, (0.2, 0.8, 1.0))       # blue
    recon_pcd = to_o3d(recon_aligned, (1.0, 0.4, 0.4)) # red

    win.add_geometry("original", orig_pcd)
    win.add_geometry("reconstruction_aligned", recon_pcd)
    win.reset_camera_to_default()

    app.add_window(win)
    app.run()


if __name__ == "__main__":
    main()
