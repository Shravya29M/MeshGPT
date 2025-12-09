import torch
import sys
import os
import numpy as np
import open3d as o3d
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh_deformation.edgeconv_model import EdgeConvDeformationNet
from mesh_deformation.build_graph import build_edges
from mesh_deformation.dataset_loader import load_single_sample


def visualize_prediction_heatmap(
    checkpoint_path="checkpoints_deformation/model_epoch100.pt",
    sample_path="data/deformation_dataset/test/sample_0000.pt"
):
    """Show model prediction and target deformation as heatmaps."""

    # Load sample
    verts_in, faces, verts_target = load_single_sample(sample_path)
    meta = torch.load(sample_path)
    deform_type = meta.get("deformation_type", "unknown")
    category = meta.get("category", "unknown")

    device = "cpu"
    verts_in = verts_in.to(device)
    faces = faces.to(device)
    verts_target = verts_target.to(device)

    # Load model
    model = EdgeConvDeformationNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Build edges
    edge_index = build_edges(verts_in, faces=faces).to(device)

    # Predict
    with torch.no_grad():
        verts_pred, _ = model(verts_in, edge_index)

    deform_pred = verts_pred - verts_in
    mag_pred = torch.norm(deform_pred, dim=1)

    deform_gt = verts_target - verts_in
    mag_gt = torch.norm(deform_gt, dim=1)

    max_val = max(mag_gt.max(), mag_pred.max()).item()
    mag_pred_n = (mag_pred / max_val).cpu().numpy()
    mag_gt_n = (mag_gt / max_val).cpu().numpy()

    try:
        cmap = cm.colormaps['jet']
    except (AttributeError, KeyError):
        try:
            cmap = cm.get_cmap('jet')
        except AttributeError:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap('jet')
    
    colors_pred = cmap(mag_pred_n)[:, :3]
    colors_gt = cmap(mag_gt_n)[:, :3]

    # Convert meshes
    mesh_in = o3d.geometry.TriangleMesh()
    mesh_in.vertices = o3d.utility.Vector3dVector(verts_in.cpu().numpy())
    mesh_in.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
    mesh_in.compute_vertex_normals()
    mesh_in.paint_uniform_color([0.7, 0.7, 0.7])
    mesh_in.translate([-2.2, 0, 0])

    mesh_pred = o3d.geometry.TriangleMesh()
    mesh_pred.vertices = o3d.utility.Vector3dVector(verts_pred.cpu().numpy())
    mesh_pred.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
    mesh_pred.compute_vertex_normals()
    mesh_pred.vertex_colors = o3d.utility.Vector3dVector(colors_pred)

    mesh_gt = o3d.geometry.TriangleMesh()
    mesh_gt.vertices = o3d.utility.Vector3dVector(verts_target.cpu().numpy())
    mesh_gt.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
    mesh_gt.compute_vertex_normals()
    mesh_gt.vertex_colors = o3d.utility.Vector3dVector(colors_gt)
    mesh_gt.translate([2.2, 0, 0])

    print(f"Category: {category}")
    print(f"Deformation: {deform_type}")
    print(f"GT max deform: {mag_gt.max().item():.4f}")
    print(f"Pred max deform: {mag_pred.max().item():.4f}")
    print("\nLEFT: INPUT (gray)")
    print("CENTER: PREDICTED (heatmap)")
    print("RIGHT: TARGET (heatmap)")
    print("Color scale: blue → yellow → red (low → high deformation)")

    o3d.visualization.draw_geometries(
        [mesh_in, mesh_pred, mesh_gt],
        window_name="INPUT (left) | PREDICTED (center) | TARGET (right)",
        width=1800,
        height=900,
        mesh_show_wireframe=False,
        mesh_show_back_face=True
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', default='data/deformation_dataset/test/sample_0000.pt')
    parser.add_argument('--checkpoint', default='checkpoints_deformation/model_epoch50.pt')
    
    args = parser.parse_args()
    
    visualize_prediction_heatmap(
        checkpoint_path=args.checkpoint,
        sample_path=args.sample
    )