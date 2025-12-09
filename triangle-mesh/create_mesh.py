import numpy as np
import open3d as o3d
import trimesh
import pyglet

import pyvista as pv

def vis_pyvista(mesh, points_array, filename="reconstruction.png"):
    # Create plotter for off-screen rendering
    plotter = pv.Plotter(off_screen=True, shape=(1, 2))
    
    # Left: mesh
    plotter.subplot(0, 0)
    pv_mesh = pv.PolyData(np.asarray(mesh.vertices), 
                          np.hstack([np.full((len(mesh.triangles), 1), 3), 
                                    mesh.triangles]).astype(int))
    plotter.add_mesh(pv_mesh, color='lightblue', show_edges=True)
    plotter.add_title("Reconstructed Mesh")
    
    # Right: point cloud
    plotter.subplot(0, 1)
    point_cloud = pv.PolyData(points_array)
    plotter.add_points(point_cloud, color='red', point_size=2)
    plotter.add_title("Original Point Cloud")
    
    plotter.screenshot(filename)
    print(f"Successfully saved visualization to: {filename}")

def vis(o3d_mesh, points_array, filename="reconstruction_o3d.png"):
    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(points_array)
    pcd_vis.paint_uniform_color([1.0, 0.0, 0.0])

    bbox = o3d_mesh.get_axis_aligned_bounding_box()
    offset = bbox.get_max_extent() * 1.5 
    pcd_vis.translate((offset, 0, 0))

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    
    vis.add_geometry(o3d_mesh)
    vis.add_geometry(pcd_vis)
    
    vis.update_geometry(o3d_mesh)
    vis.poll_events()
    vis.update_renderer()
    
    vis.capture_screen_image(filename)
    vis.destroy_window()
    
    print(f"Successfully rendered and saved image to: {filename}")

def reconstruct_mesh_with_bpa(point_cloud_path, radii_list=None):
    points = np.load(point_cloud_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50)
    )
    pcd.orient_normals_consistent_tangent_plane(k=10)

    if radii_list is None:
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        
        radius = 1.5 * avg_dist
        radii_list = [radius, radius * 2, radius * 4]
        print(f"BPA Radii determined heuristically: {radii_list}")

    o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii_list)
    )
    
    mesh = trimesh.Trimesh(
        np.asarray(o3d_mesh.vertices), 
        np.asarray(o3d_mesh.triangles), 
        vertex_normals=np.asarray(o3d_mesh.vertex_normals)
    )
    
    vis_pyvista(o3d_mesh, points)
    
    return mesh

mesh_result = reconstruct_mesh_with_bpa("/users/cnaraya2/mesh-dl/Not-MeshGPT/exported_pointclouds/chair/chair_0902_recon.npy")
print(f"Mesh created with {len(mesh_result.vertices)} vertices and {len(mesh_result.faces)} faces.")