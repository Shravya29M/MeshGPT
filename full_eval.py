#!/usr/bin/env python3
"""
Evaluate ANY FoldingNet/TransformerFoldingAE checkpoint over the full test set.
Produces:
 - Raw + ICP-aligned Precision / Recall / F1
 - Chamfer distance
 - CSV with per-sample metrics
 - Print averages + std-dev
Fully compatible with PyTorch >= 2.6 (safe_loading enabled automatically)
"""

import os
import numpy as np
import torch
import csv
from tqdm import tqdm
import open3d as o3d

from scripts.pytorch_ds import ModelNet10PC
from scripts.transformer_folding_ae import TransformerFoldingAE, chamfer_distance
from metrics import precision_recall_f1


# ============================================================
# DEVICE
# ============================================================
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# ICP ALIGNMENT
# ============================================================
def icp_align(src_np, tgt_np, threshold=0.05):
    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()

    src.points = o3d.utility.Vector3dVector(src_np)
    tgt.points = o3d.utility.Vector3dVector(tgt_np)

    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    src.transform(reg.transformation)
    return np.asarray(src.points)


# ============================================================
# SAFE CHECKPOINT LOADING (fixes PyTorch 2.6+ issues)
# ============================================================
def load_checkpoint_safely(model, ckpt_path, device):
    import torch.serialization

    # allow numpy scalar globals (fix for Taichi saved checkpoints)
    torch.serialization.add_safe_globals([
        np.float64, np.float32, np.float16,
        np.int64, np.int32, np.int16,
        np.core.multiarray.scalar
    ])

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(raw, dict) and "model_state_dict" in raw:
        model.load_state_dict(raw["model_state_dict"])
    else:
        model.load_state_dict(raw)

    print(f"✓ Successfully loaded checkpoint: {ckpt_path}")


# ============================================================
# MAIN EVALUATION LOOP
# ============================================================
def evaluate_full_model(ckpt_path, batch_size=1, csv_path="eval_results.csv"):
    device = get_device()
    print("\n===================================================")
    print(f" Evaluating checkpoint: {ckpt_path}")
    print(f" Device: {device}")
    print("===================================================\n")

    # Load model
    model = TransformerFoldingAE(num_points=2048, latent_dim=256).to(device)
    load_checkpoint_safely(model, ckpt_path, device)
    model.eval()

    # Load dataset
    test_ds = ModelNet10PC("data/modelnet10_pc_2048", split="test")
    print(f"Test samples: {len(test_ds)}")

    # Output CSV
    f = open(csv_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "index",
        "raw_precision", "raw_recall", "raw_f1",
        "icp_precision", "icp_recall", "icp_f1",
        "chamfer"
    ])

    all_raw_f1, all_icp_f1 = [], []
    all_raw_p, all_raw_r = [], []
    all_icp_p, all_icp_r = [], []
    all_chamfer = []

    # Loop over test samples
    for idx in tqdm(range(len(test_ds))):
        original = test_ds[idx]
        inp = torch.tensor(original).unsqueeze(0).float().to(device)

        with torch.no_grad():
            recon, _ = model(inp)
        recon_np = recon.squeeze(0).cpu().numpy()

        # Raw metrics
        orig_t = torch.tensor(original).float()
        recon_t = torch.tensor(recon_np).float()
        rp, rr, rf = precision_recall_f1(recon_t, orig_t)

        # Chamfer (raw)
        cd = chamfer_distance(
            orig_t.unsqueeze(0), recon_t.unsqueeze(0)
        ).item()

        # ICP alignment
        recon_icp = icp_align(recon_np, original)
        recon_icp_t = torch.tensor(recon_icp).float()
        ip, ir, iff = precision_recall_f1(recon_icp_t, orig_t)

        # Collect
        all_raw_p.append(rp)
        all_raw_r.append(rr)
        all_raw_f1.append(rf)
        all_icp_p.append(ip)
        all_icp_r.append(ir)
        all_icp_f1.append(iff)
        all_chamfer.append(cd)

        # Write CSV
        writer.writerow([idx, rp, rr, rf, ip, ir, iff, cd])

    f.close()

    print("\n==================== FINAL AVERAGES ====================")
    print(f"RAW:")
    print(f"  Precision: {np.mean(all_raw_p):.4f} ± {np.std(all_raw_p):.4f}")
    print(f"  Recall:    {np.mean(all_raw_r):.4f} ± {np.std(all_raw_r):.4f}")
    print(f"  F1:        {np.mean(all_raw_f1):.4f} ± {np.std(all_raw_f1):.4f}")
    print("\nICP-ALIGNED:")
    print(f"  Precision: {np.mean(all_icp_p):.4f} ± {np.std(all_icp_p):.4f}")
    print(f"  Recall:    {np.mean(all_icp_r):.4f} ± {np.std(all_icp_r):.4f}")
    print(f"  F1:        {np.mean(all_icp_f1):.4f} ± {np.std(all_icp_f1):.4f}")
    print("\nChamfer Distance:")
    print(f"  Mean CD:   {np.mean(all_chamfer):.6f} ± {np.std(all_chamfer):.6f}")

    print("\nResults saved to:", csv_path)
    print("========================================================\n")

    return {
        "raw_f1": np.mean(all_raw_f1),
        "icp_f1": np.mean(all_icp_f1),
        "raw_precision": np.mean(all_raw_p),
        "raw_recall": np.mean(all_raw_r),
        "icp_precision": np.mean(all_icp_p),
        "icp_recall": np.mean(all_icp_r),
        "chamfer_mean": np.mean(all_chamfer),
    }


# ============================================================
if __name__ == "__main__":
    # EXAMPLE USAGE:
    evaluate_full_model(
        ckpt_path="checkpoints_taichi_pc/best_model.pth",
        csv_path="eval_taichi_full.csv"
    )
