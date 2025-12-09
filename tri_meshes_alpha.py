import os
import numpy as np
import open3d as o3d


<<<<<<< HEAD
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
POINTCLOUD_ROOT = os.path.join(REPO_ROOT, "exported_pointclouds")
OUTPUT_ROOT = os.path.join(REPO_ROOT, "exported_meshes_alpha") 
=======
REPO_ROOT = "/Users/aditikannan/Documents/Github/Not-MeshGPT"
POINTCLOUD_ROOT = os.path.join(REPO_ROOT, "exported_pointclouds/taichi_output")
OUTPUT_ROOT = os.path.join(REPO_ROOT, "exported_meshes_alpha/taichi_output")  # Changed output folder name
>>>>>>> ad96b4dc3e1704c3a4f78947c299a71084707737

print(f"REPO_ROOT set to: {REPO_ROOT}")
print(f"POINTCLOUD_ROOT set to: {POINTCLOUD_ROOT}")
print(f"OUTPUT_ROOT set to: {OUTPUT_ROOT}\n")


def pointcloud_to_alpha_mesh(points, alpha=0.09):
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) points, got shape {points.shape}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    print(f"Original points: {len(points)}")
    if len(points) > 5000:
        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        print(f"Downsampled to: {len(pcd.points)} points")

    print(f"Creating Alpha Shape mesh with alpha={alpha}...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd,alpha=alpha)
    print(f"Mesh created!")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.75, 0.8])

    return mesh


def save_mesh_with_wireframe(mesh, out_base_path):
    os.makedirs(os.path.dirname(out_base_path), exist_ok=True)
    ply_path = out_base_path + ".ply"
    o3d.io.write_triangle_mesh(ply_path, mesh, write_vertex_colors=True)
    print(f"Saved PLY mesh to {ply_path}")
    obj_path = out_base_path + ".obj"
    o3d.io.write_triangle_mesh(obj_path, mesh, write_vertex_colors=True)
    print(f"Saved OBJ mesh to {obj_path}")
    mtl_path = out_base_path + ".mtl"
    mtl_name = os.path.basename(out_base_path)
    with open(mtl_path, 'w') as f:
        f.write(f"newmtl {mtl_name}\n")
        f.write("Ka 0.7 0.75 0.8\n")  
        f.write("Kd 0.7 0.75 0.8\n")  
        f.write("Ks 0.3 0.3 0.3\n")   
        f.write("Ns 10.0\n")          
        f.write("d 0.85\n")           
        f.write("illum 2\n")          
    
    with open(obj_path, 'r') as f:
        obj_content = f.read()
    
    if 'mtllib' not in obj_content:
        with open(obj_path, 'w') as f:
            f.write(f"mtllib {os.path.basename(mtl_path)}\n")
            f.write(f"usemtl {mtl_name}\n")
            f.write(obj_content)
    
    print(f"Saved MTL file to {mtl_path}")


def visualize_mesh_with_wireframe(mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.mesh_show_wireframe = True
    opt.mesh_show_back_face = True
    opt.line_width = 1.0
    
    vis.run()
    vis.destroy_window()


def process_pointcloud_file(pc_path, out_base_path,alpha=0.09,visualize=False):
    
    arr = np.load(pc_path)
    print(f"array shape{arr.shape}")
    if arr.ndim == 2:
        if arr.shape[1] == 3:
            points = arr
        elif arr.shape[0] == 3:
            points = arr.T
        else:
            raise ValueError(f"change array shape {arr.shape} in {pc_path}")
    else:
        raise ValueError(f"change array shape {arr.shape} in {pc_path}")

    print(f"Points ready: {points.shape}")
    
    mesh = pointcloud_to_alpha_mesh(
        points,
        alpha=alpha
    )

    save_mesh_with_wireframe(mesh, out_base_path)
    
    print(f"Triangles: {len(mesh.triangles)}")
    if visualize:
        visualize_mesh_with_wireframe(mesh)


def main():
    if not os.path.isdir(POINTCLOUD_ROOT):
        raise FileNotFoundError(f"Point cloud root not found: {POINTCLOUD_ROOT}")

    print("Point cloud root:", POINTCLOUD_ROOT)
    print("Output mesh root:", OUTPUT_ROOT)
<<<<<<< HEAD
    for category in sorted(os.listdir(POINTCLOUD_ROOT)):
        cat_dir = os.path.join(POINTCLOUD_ROOT, category)
        if not os.path.isdir(cat_dir):
            continue
        print(f"\n=== Category: {category} ===")
        out_cat_dir = os.path.join(OUTPUT_ROOT, category)
        os.makedirs(out_cat_dir, exist_ok=True)

        npy_files = sorted(
            f for f in os.listdir(cat_dir)
            if f.endswith(".npy")
        )

         for fname in npy_files:
            pc_path = os.path.join(cat_dir, fname)
            base, _ = os.path.splitext(fname)
            out_base = os.path.join(out_cat_dir, base)

            process_pointcloud_file(
                pc_path,
                out_base,
                alpha=0.09,  #smaller = more detail, larger = smoother
                visualize=False 
            )

    print("\nAll meshes generated")
=======
    print("\n*** Using Alpha Shapes (NOT Poisson) ***")
    print("Alpha Shapes only triangulates existing points - no surface hallucination!\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    npy_files = sorted(
        f for f in os.listdir(POINTCLOUD_ROOT)
        if f.endswith(".npy")
    )

    if not npy_files:
        print("No .npy files found in folder!")
        return

    for fname in npy_files:
        pc_path = os.path.join(POINTCLOUD_ROOT, fname)
        base, _ = os.path.splitext(fname)
        out_base = os.path.join(OUTPUT_ROOT, base)

        process_pointcloud_file(
            pc_path,
            out_base,
            alpha=0.09,
            visualize=False
        )

    print("\nAll Alpha Shape meshes generated!")
    print("Check folder:", OUTPUT_ROOT)


    
    print("\nTIP: If meshes have holes, try increasing alpha (e.g., 0.05 or 0.1)")
    print("     If meshes are too smooth, try decreasing alpha (e.g., 0.01 or 0.02)")
    print("\nTo view with wireframe in Open3D, use:")
    print("  mesh = o3d.io.read_triangle_mesh('your_mesh.ply')")
    print("  o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)")


>>>>>>> ad96b4dc3e1704c3a4f78947c299a71084707737
if __name__ == "__main__":
    main()