
import numpy as np
import open3d as o3d

path = "/Users/shravyamunugala/Documents/Github/Not-MeshGPT/exported_pointclouds/chair/chair_0470_original.npy"

pts = np.load(path)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)

o3d.visualization.draw_geometries([pcd])
