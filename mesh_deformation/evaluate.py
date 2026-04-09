import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh_deformation.edgeconv_model import EdgeConvDeformationNet
from mesh_deformation.build_graph import build_edges
from mesh_deformation.dataset_loader import load_single_sample

def evaluate_all_test_samples(
    checkpoint_path="checkpoints_deformation/model_epoch50.pt",
    test_dir="data/deformation_dataset/test"
):
    """Evaluate model on all test samples"""
    
    device = "cpu"
    
    # Load model
    model = EdgeConvDeformationNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get all test samples
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.pt')])
    
    print(f"EVALUATING {len(test_files)} TEST SAMPLES")
    
    errors_by_deform = {}
    errors_by_category = {}
    all_errors = []
    
    for test_file in test_files:
        sample_path = os.path.join(test_dir, test_file)
        
        # Load sample
        verts_in, faces, verts_target = load_single_sample(sample_path)
        data = torch.load(sample_path)
        
        deform_type = data.get("deformation_type", "unknown")
        category = data.get("category", "unknown")
        
        verts_in = verts_in.to(device)
        verts_target = verts_target.to(device)
        faces = faces.to(device)
        
        # Build edges and predict
        edge_index = build_edges(verts_in, faces=faces).to(device)
        
        with torch.no_grad():
            verts_pred, _ = model(verts_in, edge_index)
        
        # Calculate error
        error = torch.mean(torch.norm(verts_pred - verts_target, dim=1)).item()
        all_errors.append(error)
        
        # Track by deformation type
        if deform_type not in errors_by_deform:
            errors_by_deform[deform_type] = []
        errors_by_deform[deform_type].append(error)
        
        # Track by category
        if category not in errors_by_category:
            errors_by_category[category] = []
        errors_by_category[category].append(error)
        
        print(f"{test_file}: {error:.6f} ({category}, {deform_type})")
    
    print("SUMMARY")
    
    print(f"Overall:")
    print(f"  Mean Error: {sum(all_errors)/len(all_errors):.6f}")
    print(f"  Min Error:  {min(all_errors):.6f}")
    print(f"  Max Error:  {max(all_errors):.6f}")
    
    print(f"\nBy Deformation Type:")
    for deform_type, errors in sorted(errors_by_deform.items()):
        avg = sum(errors) / len(errors)
        print(f"  {deform_type:15s}: {avg:.6f} (n={len(errors)})")
    
    print(f"\nBy Object Category:")
    for category, errors in sorted(errors_by_category.items()):
        avg = sum(errors) / len(errors)
        print(f"  {category:15s}: {avg:.6f} (n={len(errors)})")
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints_deformation/model_epoch50.pt')
    parser.add_argument('--test_dir', default='data/deformation_dataset/test')
    
    args = parser.parse_args()
    
    evaluate_all_test_samples(args.checkpoint, args.test_dir)