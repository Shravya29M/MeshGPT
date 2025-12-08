import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys
import os
import numpy as np
import open3d as o3d  

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh_deformation.edgeconv_model import EdgeConvDeformationNet
from mesh_deformation.build_graph import build_edges
from mesh_deformation.dataset_loader import load_single_sample


def plot_mesh(ax, verts, faces, color="blue", alpha=0.4):
    """Draw a mesh on 3D axis"""
    verts = verts.cpu().numpy()
    faces = faces.cpu().numpy()
    
    triangles = verts[faces]
    mesh = Poly3DCollection(triangles, facecolor=color, edgecolor="k", alpha=alpha)
    ax.add_collection3d(mesh)
    
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1,1,1])


def plot_deformation_arrows(ax, verts_before, verts_after, subsample=30, scale=1.0, color='yellow'):
    """Draw arrows showing vertex movement"""
    verts_before = verts_before.cpu().numpy()
    verts_after = verts_after.cpu().numpy()
    
    indices = np.arange(0, len(verts_before), len(verts_before) // subsample)
    
    for i in indices[:subsample]:
        start = verts_before[i]
        end = verts_after[i]
        displacement = end - start
        
        if np.linalg.norm(displacement) > 0.001:
            ax.quiver(
                start[0], start[1], start[2],
                displacement[0] * scale, displacement[1] * scale, displacement[2] * scale,
                color=color, arrow_length_ratio=0.3, linewidth=2, alpha=0.8
            )


def compare_deformations(
    checkpoint_path="checkpoints_deformation/model_epoch50.pt",
    sample_path="data/deformation_dataset/test/sample_0000.pt"
):
    """Visualize predictions vs targets"""
    
    # Load sample
    verts_in, faces, verts_target = load_single_sample(sample_path)
    
    # Load metadata
    data = torch.load(sample_path)
    deform_type = data.get("deformation_type", "unknown")
    category = data.get("category", "unknown")
    
    device = "cpu"
    verts_in = verts_in.to(device)
    verts_target = verts_target.to(device)
    faces = faces.to(device)
    
    # Load model
    model = EdgeConvDeformationNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Build edges
    edge_index = build_edges(verts_in, faces=faces).to(device)
    
    # Predict
    with torch.no_grad():
        verts_pred, _ = model(verts_in, edge_index)
    
    # Calculate error
    error = torch.mean(torch.norm(verts_pred - verts_target, dim=1)).item()
    
    print(f"\n Category: {category}")
    print(f" Deformation: {deform_type}")
    print(f" Vertex error: {error:.6f}\n")
    
    # Visualize
    fig = plt.figure(figsize=(24, 6))
    
    ax1 = fig.add_subplot(141, projection='3d')
    plot_mesh(ax1, verts_in, faces, color='blue', alpha=0.7)
    ax1.set_title(f'Input\n({category})', fontsize=16)
    
    ax2 = fig.add_subplot(142, projection='3d')
    plot_mesh(ax2, verts_pred, faces, color='red', alpha=0.7)
    ax2.set_title('Predicted', fontsize=16)
    
    ax3 = fig.add_subplot(143, projection='3d')
    plot_mesh(ax3, verts_target, faces, color='green', alpha=0.7)
    ax3.set_title(f'Target\n({deform_type})', fontsize=16)
    
    ax4 = fig.add_subplot(144, projection='3d')
    plot_mesh(ax4, verts_in, faces, color='lightblue', alpha=0.3)
    plot_deformation_arrows(ax4, verts_in, verts_target, subsample=30, scale=1.0, color='yellow')
    ax4.set_title('Deformation Vectors', fontsize=16)
    
    plt.tight_layout()
    plt.show()


def visualize_with_deformation_heatmap(
    checkpoint_path="checkpoints_deformation/model_epoch50.pt",
    sample_path="data/deformation_dataset/test/sample_0000.pt"
):
    """Show deformation magnitude as a color gradient on the mesh"""
    
    # Load sample
    verts_in, faces, verts_target = load_single_sample(sample_path)
    
    # Load metadata
    data = torch.load(sample_path)
    deform_type = data.get("deformation_type", "unknown")
    category = data.get("category", "unknown")
    
    device = "cpu"
    verts_in = verts_in.to(device)
    verts_target = verts_target.to(device)
    faces = faces.to(device)
    
    # Load model
    model = EdgeConvDeformationNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Build edges
    edge_index = build_edges(verts_in, faces=faces).to(device)
    
    # Predict
    with torch.no_grad():
        verts_pred, _ = model(verts_in, edge_index)
    
    # Calculate per-vertex deformation magnitude
    deformation = verts_target - verts_in  # How much each vertex moved
    deformation_magnitude = torch.norm(deformation, dim=1)  # Distance moved
    
    # Normalize to 0-1 range
    max_deform = deformation_magnitude.max()
    min_deform = deformation_magnitude.min()
    
    if max_deform > min_deform:
        normalized_deform = (deformation_magnitude - min_deform) / (max_deform - min_deform)
    else:
        normalized_deform = torch.zeros_like(deformation_magnitude)
    
    # Convert to numpy
    verts_target_np = verts_target.cpu().numpy()
    faces_np = faces.cpu().numpy()
    normalized_deform_np = normalized_deform.cpu().numpy()
    
    # Create colormap: Blue (no deformation) -> Yellow -> Red (max deformation)
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    colormap = cm.get_cmap('jet')  # Blue -> Cyan -> Yellow -> Red
    colors = colormap(normalized_deform_np)[:, :3]  # RGB only
    
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_target_np)
    mesh.triangles = o3d.utility.Vector3iVector(faces_np)
    mesh.compute_vertex_normals()
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Print info
    avg_error = torch.mean(torch.norm(verts_pred - verts_target, dim=1)).item()
    
    print(f"\n{'='*60}")
    print(f" Category: {category}")
    print(f" Deformation: {deform_type}")
    print(f"Average error: {avg_error:.6f}")
    print(f"Max deformation: {max_deform.item():.6f}")
    
    print("\nColor Coding:")
    print("   Blue = No deformation")
    print("   Yellow = Medium deformation")
    print("   Red = Maximum deformation\n")
    
    # Visualize
    o3d.visualization.draw_geometries(
        [mesh],
        window_name=f"Deformation Heatmap - {category} ({deform_type})",
        width=1200,
        height=900,
        mesh_show_wireframe=True,
        mesh_show_back_face=True
    )


def visualize_side_by_side_with_heatmap(
    checkpoint_path="checkpoints_deformation/model_epoch50.pt",
    sample_path="data/deformation_dataset/test/sample_0000.pt"
):
    """Show original (gray) next to deformed (heat-mapped)"""
    
    # Load sample
    verts_in, faces, verts_target = load_single_sample(sample_path)
    
    # Load metadata
    data = torch.load(sample_path)
    deform_type = data.get("deformation_type", "unknown")
    category = data.get("category", "unknown")
    
    device = "cpu"
    verts_in = verts_in.to(device)
    verts_target = verts_target.to(device)
    faces = faces.to(device)
    
    # Calculate deformation magnitude
    deformation = verts_target - verts_in
    deformation_magnitude = torch.norm(deformation, dim=1)
    
    max_deform = deformation_magnitude.max()
    min_deform = deformation_magnitude.min()
    
    if max_deform > min_deform:
        normalized_deform = (deformation_magnitude - min_deform) / (max_deform - min_deform)
    else:
        normalized_deform = torch.zeros_like(deformation_magnitude)
    
    # Convert to numpy
    verts_in_np = verts_in.cpu().numpy()
    verts_target_np = verts_target.cpu().numpy()
    faces_np = faces.cpu().numpy()
    normalized_deform_np = normalized_deform.cpu().numpy()
    
    # Original mesh (gray)
    mesh_original = o3d.geometry.TriangleMesh()
    mesh_original.vertices = o3d.utility.Vector3dVector(verts_in_np)
    mesh_original.triangles = o3d.utility.Vector3iVector(faces_np)
    mesh_original.compute_vertex_normals()
    mesh_original.paint_uniform_color([0.7, 0.7, 0.7])  # Gray
    mesh_original.translate([-2.0, 0, 0])
    
    # Deformed mesh (heat-mapped)
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    colormap = cm.get_cmap('jet')
    colors = colormap(normalized_deform_np)[:, :3]
    
    mesh_deformed = o3d.geometry.TriangleMesh()
    mesh_deformed.vertices = o3d.utility.Vector3dVector(verts_target_np)
    mesh_deformed.triangles = o3d.utility.Vector3iVector(faces_np)
    mesh_deformed.compute_vertex_normals()
    mesh_deformed.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh_deformed.translate([2.0, 0, 0])
    
    print(f"\n{'='*60}")
    print(f" Category: {category}")
    print(f" Deformation: {deform_type}")
    print(f" Max deformation: {max_deform.item():.6f}")
    
    print("\n Left (Gray): Original")
    print("   Right (Color): Deformed (colored by magnitude)\n")
    
    o3d.visualization.draw_geometries(
        [mesh_original, mesh_deformed],
        window_name=f"Before/After - {category} ({deform_type})",
        width=1600,
        height=900,
        mesh_show_wireframe=True,
        mesh_show_back_face=True
    )



def visualize_single_mesh_open3d(
    checkpoint_path="checkpoints_deformation/model_epoch50.pt",
    sample_path="data/deformation_dataset/test/sample_0000.pt",
    show_type="predicted"
):
    """Show single mesh in detail"""
    
    verts_in, faces, verts_target = load_single_sample(sample_path)
    data = torch.load(sample_path)
    deform_type = data.get("deformation_type", "unknown")
    category = data.get("category", "unknown")
    
    device = "cpu"
    verts_in = verts_in.to(device)
    verts_target = verts_target.to(device)
    faces = faces.to(device)
    
    model = EdgeConvDeformationNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    edge_index = build_edges(verts_in, faces=faces).to(device)
    
    with torch.no_grad():
        verts_pred, _ = model(verts_in, edge_index)
    
    # Select mesh
    if show_type == "input":
        verts_show = verts_in
        color = [0.2, 0.4, 0.8]
        title = f"Input - {category}"
    elif show_type == "target":
        verts_show = verts_target
        color = [0.2, 0.8, 0.2]
        title = f"Target - {category} ({deform_type})"
    else:  # predicted
        verts_show = verts_pred
        color = [0.8, 0.2, 0.2]
        title = f"Predicted - {category}"
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_show.cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    
    o3d.visualization.draw_geometries(
        [mesh],
        window_name=title,
        width=1200,
        height=900,
        mesh_show_wireframe=True,
        mesh_show_back_face=True
    )



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='matplotlib', 
                       choices=['matplotlib', 'open3d', 'single', 'heatmap', 'sidebyside'])
    parser.add_argument('--sample', default='data/deformation_dataset/test/sample_0000.pt')
    parser.add_argument('--checkpoint', default='checkpoints_deformation/model_epoch50.pt')
    parser.add_argument('--show', default='predicted', choices=['input', 'predicted', 'target'])
    
    args = parser.parse_args()
    
    if args.mode == 'matplotlib':
        compare_deformations(
            checkpoint_path=args.checkpoint,
            sample_path=args.sample
        )
    elif args.mode == 'open3d':
        visualize_with_open3d(
            checkpoint_path=args.checkpoint,
            sample_path=args.sample
        )
    elif args.mode == 'single':
        visualize_single_mesh_open3d(
            checkpoint_path=args.checkpoint,
            sample_path=args.sample,
            show_type=args.show
        )
    elif args.mode == 'heatmap':
        visualize_with_deformation_heatmap(
            checkpoint_path=args.checkpoint,
            sample_path=args.sample
        )
    elif args.mode == 'sidebyside':
        visualize_side_by_side_with_heatmap(
            checkpoint_path=args.checkpoint,
            sample_path=args.sample
        )