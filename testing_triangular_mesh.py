import open3d as o3d
mesh = o3d.io.read_triangle_mesh("/Users/shravyamunugala/Documents/Github/Not-MeshGPT/exported_meshes_alpha/chair/chair_0902_recon.ply")
o3d.visualization.draw_geometries([mesh], 
                                  mesh_show_wireframe=True,
                                  mesh_show_back_face=True)