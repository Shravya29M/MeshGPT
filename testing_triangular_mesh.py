import open3d as o3d
import os
repo_root = os.path.dirname(os.path.abspath(__file__))  
mesh_path = os.path.join(repo_root, "exported_meshes_alpha/chair/chair_0183_original.ply")
mesh = o3d.io.read_triangle_mesh(mesh_path)
o3d.visualization.draw_geometries([mesh], 
                                  mesh_show_wireframe=True,
                                  mesh_show_back_face=True)