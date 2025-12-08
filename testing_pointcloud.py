
import numpy as np
import open3d as o3d
import os
repo_root = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(repo_root, "exported_pointclouds/chair/chair_0470_original.npy")
pts = np.load(path)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)

o3d.visualization.draw_geometries([pcd])
