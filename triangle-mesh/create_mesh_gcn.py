import torch
import numpy as np
from scipy.spatial import cKDTree
import mcubes 
import pyvista as pv
#from create_mesh import vis_pyvista 
import trimesh

def vis_pyvista(mesh, points_array, filename="reconstruction.png"):
    # Create plotter for off-screen rendering
    plotter = pv.Plotter(off_screen=True, shape=(1, 2))
    
    # Left: mesh
    plotter.subplot(0, 0)
    pv_mesh = pv.PolyData(np.asarray(mesh.vertices), 
                          np.hstack([np.full((len(mesh.faces), 1), 3), 
                                    mesh.faces]).astype(int))
    plotter.add_mesh(pv_mesh, color='lightblue', show_edges=True)
    plotter.add_title("Reconstructed Mesh")
    
    # Right: point cloud
    plotter.subplot(0, 1)
    point_cloud = pv.PolyData(points_array)
    plotter.add_points(point_cloud, color='red', point_size=2)
    plotter.add_title("Original Point Cloud")
    
    plotter.screenshot(filename)
    print(f"Successfully saved visualization to: {filename}")
class PointCloudToSDF:
    """Convert point cloud to mesh via SDF"""
    def __init__(self, points, normals=None):
        self.points = points
        self.tree = cKDTree(points)
        
        # Estimate normals if not provided
        if normals is None:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            self.normals = np.asarray(pcd.normals)
        else:
            self.normals = normals
    
    def compute_sdf(self, query_points):
        """Compute signed distance for query points"""
        # Find nearest point in cloud
        distances, indices = self.tree.query(query_points)
        
        # Get normal at nearest point
        nearest_normals = self.normals[indices]
        
        # Compute signed distance using normal direction
        vectors = query_points - self.points[indices]
        signs = np.sign((vectors * nearest_normals).sum(axis=1))
        
        return signs * distances
    
    def extract_mesh(self, resolution=256, padding=0.1):
        """Extract mesh using marching cubes"""
        # Create grid
        bounds_min = self.points.min(axis=0) - padding
        bounds_max = self.points.max(axis=0) + padding
        
        x = np.linspace(bounds_min[0], bounds_max[0], resolution)
        y = np.linspace(bounds_min[1], bounds_max[1], resolution)
        z = np.linspace(bounds_min[2], bounds_max[2], resolution)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        
        # Compute SDF on grid
        sdf_values = self.compute_sdf(grid_points)
        sdf_grid = sdf_values.reshape(resolution, resolution, resolution)
        
        # Extract mesh using marching cubes
        vertices, triangles = mcubes.marching_cubes(sdf_grid, 0)
        
        # Scale vertices back to original coordinates
        vertices = vertices / (resolution - 1)  # Normalize to [0, 1]
        vertices = vertices * (bounds_max - bounds_min) + bounds_min
        
        return trimesh.Trimesh(vertices=vertices, faces=triangles)


# Usage
points = np.load("/users/cnaraya2/mesh-dl/Not-MeshGPT/exported_pointclouds/chair/chair_0470_recon.npy")
sdf_model = PointCloudToSDF(points)
mesh = sdf_model.extract_mesh(resolution=128)
vis_pyvista(mesh,points)