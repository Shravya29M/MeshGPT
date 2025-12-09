import torch
import numpy as np
import open3d as o3d

from scripts.pytorch_ds import ModelNet10PC
from scripts.transformer_folding_ae import TransformerFoldingAE
from metrics import precision_recall_f1


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_o3d(pc, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.paint_uniform_color(color)
    return pcd


def icp_align(src_np, tgt_np, threshold=0.05):
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
    dataset = ModelNet10PC("data/modelnet10_pc_2048", split="test")
    print("Total test samples:", len(dataset))

    idx = np.random.randint(0, len(dataset))
    print("Using index:", idx)
    original = dataset[idx]   
    model = TransformerFoldingAE(num_points=2048, latent_dim=256).to(device)
    ckpt_path = "checkpoints_foldingnet/foldingnet_epoch1.pth"  
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    with torch.no_grad():
        inp = torch.tensor(original).unsqueeze(0).to(device)
        recon, _ = model(inp)
        recon = recon.squeeze(0).cpu().numpy()

    orig_t = torch.tensor(original).float()
    recon_t = torch.tensor(recon).float()
    p0, r0, f0 = precision_recall_f1(recon_t, orig_t)
    print("\n=== Raw (Unaligned) Metrics ===")
    print("Precision:", p0)
    print("Recall:", r0)
    print("F1:", f0)

    recon_aligned = icp_align(recon, original)
    recon_align_t = torch.tensor(recon_aligned).float()
    p1, r1, f1 = precision_recall_f1(recon_align_t, orig_t)
    print("\n=== After ICP Alignment ===")
    print("Precision:", p1)
    print("Recall:", r1)
    print("F1:", f1)

    from open3d.visualization import gui
    app = gui.Application.instance
    app.initialize()

    win = o3d.visualization.O3DVisualizer( "FoldingNetAE: Original + Reconstruction (Aligned)", 1280, 720)

    orig_pcd = to_o3d(original, (0.2, 0.8, 1.0))       # blue
    recon_pcd = to_o3d(recon_aligned, (1.0, 0.4, 0.4)) # red

    win.add_geometry("original", orig_pcd)
    win.add_geometry("reconstruction_aligned", recon_pcd)
    win.reset_camera_to_default()

    app.add_window(win)
    app.run()


if __name__ == "__main__":
    main()
