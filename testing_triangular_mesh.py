import open3d as o3d
mesh = o3d.io.read_triangle_mesh("exported_meshes_alpha/taichi_output/sample_90_recon.ply")
o3d.visualization.draw_geometries([mesh], 
                                  mesh_show_wireframe=True,
                                  mesh_show_back_face=True)