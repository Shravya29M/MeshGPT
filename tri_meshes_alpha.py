#!/usr/bin/env python3

print("=" * 50)

import os
import numpy as np
import open3d as o3d


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
POINTCLOUD_ROOT = os.path.join(REPO_ROOT, "exported_pointclouds")
OUTPUT_ROOT = os.path.join(REPO_ROOT, "exported_meshes_alpha")  # Changed output folder name

print(f"REPO_ROOT set to: {REPO_ROOT}")
print(f"POINTCLOUD_ROOT set to: {POINTCLOUD_ROOT}")
print(f"OUTPUT_ROOT set to: {OUTPUT_ROOT}\n")


def pointcloud_to_alpha_mesh(points, alpha=0.09):
    """
    points: (N, 3) numpy array
    alpha: controls the level of detail (smaller = more detail, but may create holes)
           typical range: 0.01 to 0.1
    returns: Open3D TriangleMesh
    
    Alpha Shapes only creates triangles between nearby points - it does NOT
    hallucinate surfaces like Poisson. No normals needed!
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) points, got shape {points.shape}")

    # 1) Build Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    print(f"     Original points: {len(points)}")
    
    # Downsample if too many points (Alpha Shapes is VERY slow with many points)
    if len(points) > 5000:
        print(f"     Downsampling... (this helps performance)")
    # if len(points) > 20000:
    #     pcd = pcd.voxel_down_sample(voxel_size=0.01)

        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        print(f"     Downsampled to: {len(pcd.points)} points")

    # 2) Create mesh using Alpha Shapes
    print(f"     Creating Alpha Shape mesh with alpha={alpha}...")
    print(f"     (This may take 30-60 seconds for complex shapes...)")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, 
        alpha=alpha
    )
    print(f"     Mesh created!")

    # 3) Cleanup
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # 4) Compute normals for better visualization
    mesh.compute_vertex_normals()
    
    # 5) Paint with a light color for better visibility
    mesh.paint_uniform_color([0.7, 0.75, 0.8])  # light blue-grey

    return mesh


def save_mesh_with_wireframe(mesh, out_base_path):
    """
    Save mesh in formats that support wireframe visualization.
    Creates both PLY and OBJ files, plus a visualization-ready version.
    """
    os.makedirs(os.path.dirname(out_base_path), exist_ok=True)
    
    # Save PLY (best for Open3D visualization)
    ply_path = out_base_path + ".ply"
    o3d.io.write_triangle_mesh(ply_path, mesh, write_vertex_colors=True)
    print(f"     Saved PLY mesh -> {ply_path}")

    # Save OBJ with MTL for better material support
    obj_path = out_base_path + ".obj"
    o3d.io.write_triangle_mesh(obj_path, mesh, write_vertex_colors=True)
    print(f"     Saved OBJ mesh -> {obj_path}")
    
    # Create a companion MTL file for better rendering
    mtl_path = out_base_path + ".mtl"
    mtl_name = os.path.basename(out_base_path)
    with open(mtl_path, 'w') as f:
        f.write(f"newmtl {mtl_name}\n")
        f.write("Ka 0.7 0.75 0.8\n")  # Ambient color
        f.write("Kd 0.7 0.75 0.8\n")  # Diffuse color
        f.write("Ks 0.3 0.3 0.3\n")   # Specular color
        f.write("Ns 10.0\n")          # Specular exponent
        f.write("d 0.85\n")           # Transparency (0.85 = slightly transparent)
        f.write("illum 2\n")          # Illumination model
    
    # Update OBJ to reference MTL
    with open(obj_path, 'r') as f:
        obj_content = f.read()
    
    if 'mtllib' not in obj_content:
        with open(obj_path, 'w') as f:
            f.write(f"mtllib {os.path.basename(mtl_path)}\n")
            f.write(f"usemtl {mtl_name}\n")
            f.write(obj_content)
    
    print(f"     Saved MTL file -> {mtl_path}")


def visualize_mesh_with_wireframe(mesh):
    """
    Visualize the mesh in Open3D with wireframe overlay.
    This is for testing - you can call this function to preview.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    
    # Set render options for wireframe
    opt = vis.get_render_option()
    opt.mesh_show_wireframe = True
    opt.mesh_show_back_face = True
    opt.line_width = 1.0
    
    vis.run()
    vis.destroy_window()


def process_pointcloud_file(pc_path, out_base_path,
                            alpha=0.09,
                            visualize=False):
    """
    pc_path: path to *.npy point cloud (must be *_recon.npy)
    out_base_path: output path without extension
    alpha: Alpha Shapes parameter (smaller = more detail, larger = smoother)
    visualize: if True, show interactive preview with wireframe
    """
    print(f"\n  -> Loading {pc_path}")
    arr = np.load(pc_path)
    print(f"     Array loaded: shape {arr.shape}")

    # Handle (N,3) or (3,N)
    if arr.ndim == 2:
        if arr.shape[1] == 3:
            points = arr
        elif arr.shape[0] == 3:
            points = arr.T
        else:
            raise ValueError(f"Unsupported array shape {arr.shape} in {pc_path}")
    else:
        raise ValueError(f"Unsupported array shape {arr.shape} in {pc_path}")

    print(f"     Points ready: {points.shape}")
    
    mesh = pointcloud_to_alpha_mesh(
        points,
        alpha=alpha
    )

    save_mesh_with_wireframe(mesh, out_base_path)
    
    print(f"     Triangles: {len(mesh.triangles)}")
    print(f"    Complete!\n")
    
    # Optional: visualize in Open3D
    if visualize:
        visualize_mesh_with_wireframe(mesh)


def main():
    if not os.path.isdir(POINTCLOUD_ROOT):
        raise FileNotFoundError(f"Point cloud root not found: {POINTCLOUD_ROOT}")

    print("Point cloud root:", POINTCLOUD_ROOT)
    print("Output mesh root:", OUTPUT_ROOT)
    print("\n*** Using Alpha Shapes (NOT Poisson) ***")
    print("Alpha Shapes only triangulates existing points - no surface hallucination!\n")

    for category in sorted(os.listdir(POINTCLOUD_ROOT)):
        cat_dir = os.path.join(POINTCLOUD_ROOT, category)
        if not os.path.isdir(cat_dir):
            continue

        print(f"\n=== Category: {category} ===")
        out_cat_dir = os.path.join(OUTPUT_ROOT, category)
        os.makedirs(out_cat_dir, exist_ok=True)

        # Only use *_recon.npy files
        npy_files = sorted(
            f for f in os.listdir(cat_dir)
            if f.endswith(".npy")
        )

        if not npy_files:
            print("  (no *_recon.npy files found)")
            continue

        for fname in npy_files:
            pc_path = os.path.join(cat_dir, fname)
            base, _ = os.path.splitext(fname)
            out_base = os.path.join(out_cat_dir, base)

            process_pointcloud_file(
                pc_path,
                out_base,
                alpha=0.09,  # Adjust this: smaller = more detail, larger = smoother
                visualize=False  # Set to True to preview first mesh
            )

    print("\nAll Alpha Shape meshes generated.")
    print("\nTIP: If meshes have holes, try increasing alpha (e.g., 0.05 or 0.1)")
    print("     If meshes are too smooth, try decreasing alpha (e.g., 0.01 or 0.02)")
    print("\nTo view with wireframe in Open3D, use:")
    print("  mesh = o3d.io.read_triangle_mesh('your_mesh.ply')")
    print("  o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)")


if __name__ == "__main__":
    main()