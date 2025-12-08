import os
import sys
import torch
import numpy as np
import open3d as o3d
import glob
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)  
sys.path.insert(0, REPO_ROOT)

from deform_mesh import (
    deform_upward, deform_bump, deform_sitting, 
    deform_sag, deform_squeeze, deform_twist,
    deform_bend, deform_taper, deform_shear,
    deform_bulge, deform_stretch, deform_ripple,
    deform_dent, deform_inflate
)

def load_mesh_from_ply(ply_path):
    mesh = o3d.io.read_triangle_mesh(ply_path)
    
    # Extract vertices and faces
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    # Convert to torch
    verts = torch.from_numpy(verts).float()
    faces = torch.from_numpy(faces).long()
    
    return verts, faces


def normalize_mesh(verts):
    center = verts.mean(dim=0)
    verts = verts - center
    scale = verts.norm(dim=1).max()
    if scale > 0:
        verts = verts / scale
    return verts


def random_deformation(verts):
    choice = random.choice([
        'sitting', 'upward', 'sag', 'bump', 'squeeze',  
        'bend', 'taper', 'shear', 'bulge', 'stretch',   
        'ripple', 'dent', 'inflate'                     
    ])
    
    if choice == 'sitting':
        verts_deformed = deform_sitting(verts, compression=random.uniform(0.1, 0.2))
    elif choice == 'upward':
        verts_deformed = deform_upward(verts, amount=random.uniform(0.05, 0.15))
    elif choice == 'sag':
        verts_deformed = deform_sag(verts, amount=random.uniform(0.05, 0.12))
    elif choice == 'bump':
        verts_deformed = deform_bump(verts, center=(0, -0.2, 0), radius=0.4, 
                                     strength=random.uniform(0.08, 0.15))
   
    elif choice == 'bend':
        return deform_bend(verts, angle=random.uniform(0.2, 0.4)), choice
    elif choice == 'taper':
        return deform_taper(verts, factor=random.uniform(0.2, 0.4)), choice
    elif choice == 'shear':
        return deform_shear(verts, amount=random.uniform(0.15, 0.3)), choice
    elif choice == 'bulge':
        return deform_bulge(verts, strength=random.uniform(0.15, 0.3)), choice
    elif choice == 'stretch':
        return deform_stretch(verts, axis=random.choice(['x','y','z']), factor=random.uniform(0.2, 0.4)), choice
    elif choice == 'ripple':
        return deform_ripple(verts, frequency=random.uniform(3, 8), amplitude=random.uniform(0.05, 0.15)), choice
    elif choice == 'dent':
        return deform_dent(verts, num_dents=random.randint(2, 4)), choice
    elif choice == 'inflate':
        return deform_inflate(verts, amount=random.uniform(0.15, 0.3)), choice
    else:  # squeeze
        verts_deformed = deform_squeeze(verts, factor=random.uniform(0.05, 0.12))
    
    return verts_deformed, choice


def build_deformation_dataset(
    mesh_dir="exported_meshes_alpha",
    out_dir="data/deformation_dataset",
    samples_per_category=20,
    split="train"
):
    
    out_split_dir = os.path.join(out_dir, split)
    os.makedirs(out_split_dir, exist_ok=True)
    
    counter = 0
    category_counts = {}
    
    # Get all categories
    categories = [d for d in os.listdir(mesh_dir) 
                  if os.path.isdir(os.path.join(mesh_dir, d))]
    

    for category in sorted(categories):
        cat_path = os.path.join(mesh_dir, category)
        category_counts[category] = 0
        
        ply_files = sorted(glob.glob(os.path.join(cat_path, "*_original.ply")))
        
        if not ply_files:
            print(f"No meshes found in {category}")
            continue
        
        print(f"\n Processing {category} ({len(ply_files)} meshes available)")
        
        # Sample randomly for variety
        selected_files = random.sample(ply_files, min(samples_per_category, len(ply_files)))
        
        for ply_file in selected_files:
            filename = os.path.basename(ply_file)
            print(f"  → {filename}")
            
            try:
                # Load mesh
                verts, faces = load_mesh_from_ply(ply_file)
                
                if verts.shape[0] < 100 or verts.shape[0] > 50000:
                    print(f"Skipping: {verts.shape[0]} vertices (out of range)")
                    continue
                
                if faces.shape[0] < 50:
                    print(f"Skipping: {faces.shape[0]} faces (too few)")
                    continue
                
                verts = normalize_mesh(verts)
                
                # Apply deformation
                verts_target, deform_type = random_deformation(verts)
                
                save_path = os.path.join(out_split_dir, f"sample_{counter:04d}.pt")
                torch.save({
                    "verts_in": verts,
                    "faces": faces,
                    "verts_target": verts_target,
                    "deformation_type": deform_type,
                    "category": category,
                    "source_file": filename
                }, save_path)
                
                counter += 1
                category_counts[category] += 1
                
                print(f"Saved: {verts.shape[0]} verts, {faces.shape[0]} faces, deform: {deform_type}")
                
            except Exception as e:
                print(f" Error: {e}")
                continue
    
    print(f"{split.upper()} DATASET COMPLETE")
    
    print(f"Total samples: {counter}")
    for cat, count in sorted(category_counts.items()):
        print(f"{cat}: {count}")

    return counter


if __name__ == "__main__":
    # Build train and test splits
    
    train_count = build_deformation_dataset(
        mesh_dir="exported_meshes_alpha",
        out_dir="data/deformation_dataset",
        samples_per_category=100, 
        split="train"
    )
    
    test_count = build_deformation_dataset(
        mesh_dir="exported_meshes_alpha",
        out_dir="data/deformation_dataset",
        samples_per_category=50,   
        split="test"
    )
    
    print(f"Train samples: {train_count}")
    print(f"Test samples: {test_count}")
    print(f"Total: {train_count + test_count}")