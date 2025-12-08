# eval_foldingnet_ae.py

import torch
import numpy as np
import open3d as o3d
from torch.utils.data import DataLoader

from scripts.pytorch_ds import ModelNet10PC
from scripts.transformer_folding_ae import TransformerFoldingAE
from metrics import precision_recall_f1


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def icp_align(src_np, tgt_np, threshold=0.05):
    """Align src to tgt with point-to-point ICP (Open3D)."""
    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_np)
    tgt.points = o3d.utility.Vector3dVector(tgt_np)

    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    src.transform(reg.transformation)
    return np.asarray(src.points)


def main():
    device = get_device()
    print("Using device:", device)

    # --- Test dataset ---
    data_root = "data/modelnet10_pc_2048"
    test_ds = ModelNet10PC(data_root, split="test")
    print("Test samples:", len(test_ds))

    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    # --- Load trained model / checkpoint ---
    model = TransformerFoldingAE(num_points=2048, latent_dim=256).to(device)
    ckpt_path = "checkpoints_foldingnet/old_foldingnet_epoch80.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # --- Accumulators for averages ---
    sum_p_raw = 0.0
    sum_r_raw = 0.0
    sum_f_raw = 0.0

    sum_p_icp = 0.0
    sum_r_icp = 0.0
    sum_f_icp = 0.0

    n_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            # batch: (B, N, 3) as numpy -> convert to torch
            batch = batch.to(device)  # (B, N, 3)

            recon, _ = model(batch)   # (B, N, 3)
            recon_np = recon.cpu().numpy()
            orig_np = batch.cpu().numpy()

            B = orig_np.shape[0]

            for b in range(B):
                orig = orig_np[b]
                rec  = recon_np[b]

                # --- raw metrics ---
                orig_t = torch.tensor(orig).float()
                rec_t  = torch.tensor(rec).float()
                p0, r0, f0 = precision_recall_f1(rec_t, orig_t)

                # --- ICP metrics ---
                rec_aligned = icp_align(rec, orig)
                rec_align_t = torch.tensor(rec_aligned).float()
                p1, r1, f1 = precision_recall_f1(rec_align_t, orig_t)

                sum_p_raw += float(p0)
                sum_r_raw += float(r0)
                sum_f_raw += float(f0)

                sum_p_icp += float(p1)
                sum_r_icp += float(r1)
                sum_f_icp += float(f1)

                n_samples += 1

    # --- Averages ---
    avg_p_raw = sum_p_raw / n_samples
    avg_r_raw = sum_r_raw / n_samples
    avg_f_raw = sum_f_raw / n_samples

    avg_p_icp = sum_p_icp / n_samples
    avg_r_icp = sum_r_icp / n_samples
    avg_f_icp = sum_f_icp / n_samples

    print("\n=== AVERAGE over entire test set ===")
    print(f"Raw  -> P: {avg_p_raw:.4f} | R: {avg_r_raw:.4f} | F1: {avg_f_raw:.4f}")
    print(f"ICP  -> P: {avg_p_icp:.4f} | R: {avg_r_icp:.4f} | F1: {avg_f_icp:.4f}")


if __name__ == "__main__":
    main()
