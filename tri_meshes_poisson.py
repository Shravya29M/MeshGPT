#!/usr/bin/env python3

import os
import numpy as np
import open3d as o3d


REPO_ROOT = "/Users/shravyamunugala/Documents/Github/Not-MeshGPT"
POINTCLOUD_ROOT = os.path.join(REPO_ROOT, "exported_pointclouds")
OUTPUT_ROOT = os.path.join(REPO_ROOT, "exported_meshes_poisson")  


def pointcloud_to_poisson_mesh(points,
                               depth=8,
                               target_triangles=20000):
    """
    points: (N, 3) numpy array
    depth: Poisson octree depth (higher = more detail, slower)
    target_triangles: simplify mesh to roughly this many triangles
    returns: Open3D TriangleMesh
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) points, got shape {points.shape}")

    # 1) Build Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 2) Estimate normals (required for Poisson)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1,
            max_nn=30
        )
    )
    pcd.orient_normals_consistent_tangent_plane(10)

    # 3) Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth
    )

    # 4) Crop to original bounding box to remove far-away artifacts
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # 5) Optional: simplify to a target triangle count
    if target_triangles is not None and len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)

    # 6) Cleanup
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # 7) Make sure normals are computed
    mesh.compute_vertex_normals()
    
    # 8) Paint with a light color for better visibility
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
                            depth=8,
                            target_triangles=20000,
                            visualize=False):
    """
    pc_path: path to *.npy point cloud (must be *_recon.npy)
    out_base_path: output path without extension
    visualize: if True, show interactive preview with wireframe
    """
    print(f"  -> Loading {pc_path}")
    arr = np.load(pc_path)

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

    mesh = pointcloud_to_poisson_mesh(
        points,
        depth=depth,
        target_triangles=target_triangles
    )

    save_mesh_with_wireframe(mesh, out_base_path)
    
    print(f"     Triangles: {len(mesh.triangles)}\n")
    
    # Optional: visualize in Open3D
    if visualize:
        visualize_mesh_with_wireframe(mesh)


def main():
    if not os.path.isdir(POINTCLOUD_ROOT):
        raise FileNotFoundError(f"Point cloud root not found: {POINTCLOUD_ROOT}")

    print("Point cloud root:", POINTCLOUD_ROOT)
    print("Output mesh root:", OUTPUT_ROOT)

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
                depth=8,
                target_triangles=20000,
                visualize=False  # Set to True to preview first mesh
            )

    print("\nAll Poisson meshes generated.")
    print("\nTo view with wireframe in Open3D, use:")
    print("  mesh = o3d.io.read_triangle_mesh('your_mesh.ply')")
    print("  o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)")


if __name__ == "__main__":
    main()