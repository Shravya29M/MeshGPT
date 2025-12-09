import numpy as np
import open3d as o3d


def visualize_pointcloud_pretty(points, color=(0.2, 0.6, 1.0), point_size=3.0):
    """
    Pretty visualization for ModelNet point clouds.
    
    - Centers point cloud
    - Normalizes scale
    - Applies uniform color
    - Improves render settings
    """

    # Convert numpy → Open3D cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # ----------------------------------------------------
    # 1. Normalize (center + scale)
    # ----------------------------------------------------
    pts = points.copy()
    pts = pts - pts.mean(axis=0)               # center
    scale = np.max(np.linalg.norm(pts, axis=1))
    pts = pts / scale                          # scale to unit sphere

    pcd.points = o3d.utility.Vector3dVector(pts)

    # ----------------------------------------------------
    # 2. Uniform color (looks MUCH nicer than rainbow)
    # ----------------------------------------------------
    pcd.paint_uniform_color(color)

    # ----------------------------------------------------
    # 3. Visualizer with enhanced settings
    # ----------------------------------------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Generated Sample", width=1280, height=720)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])     # white background
    opt.point_size = point_size
    opt.show_coordinate_frame = True

    # Smooth edges & better lighting
    opt.light_on = True

    vis.run()
    vis.destroy_window()


# ========================================================
# Load your file and visualize cleanly
# ========================================================

path = "exported_pointclouds/taichi_output/sample_900_recon.npy"

pts = np.load(path)

visualize_pointcloud_pretty(
    pts,
    color=(0.1, 0.4, 0.9),     # pleasant blue
    point_size=4.0            # make points visible
)
