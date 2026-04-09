import torch
import torch.optim as optim
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mesh_deformation.dataset_loader import get_dataloader
from mesh_deformation.edgeconv_model import EdgeConvDeformationNet
from mesh_deformation.build_graph import build_edges
from mesh_deformation.deformation_loss import deformation_loss


def train(
    dataset_root="data/deformation_dataset/train",
    num_epochs=100,
    lr=1e-3,
    device=None,
    checkpoint_dir="checkpoints_deformation"
):
    """Train the deformation model"""
    
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    print("MESH DEFORMATION TRAINING")
    print(f"Device: {device}")
    print(f"Dataset: {dataset_root}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Data loader
    try:
        loader = get_dataloader(dataset_root, batch_size=1, shuffle=True)
        print(f"Loaded dataset: {len(loader)} samples\n")
    except Exception as e:
        print(f" Error loading dataset: {e}")
        print(f"Make sure you've run: python3 mesh_deformation/build_from_alpha_meshes.py")
        return
    
    # Model
    model = EdgeConvDeformationNet(
        in_dim=3,
        hidden_dim=64,
        num_layers=3,
        out_dim=3
    ).to(device)
    
    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters\n")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Edge cache
    edge_cache = {}
    
    print("Starting training...\n")
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_data in loader:
            verts_in, faces, verts_target = batch_data
            
            verts_in = verts_in.to(device)
            verts_target = verts_target.to(device)
            faces = faces.to(device)
            
            # Build/cache edges
            num_verts = verts_in.shape[0]
            num_faces = faces.shape[0] if faces.numel() > 0 else 0
            cache_key = (num_verts, num_faces)
            
            if cache_key not in edge_cache:
                edge_cache[cache_key] = build_edges(verts_in, faces=faces)
            
            edge_index = edge_cache[cache_key].to(device)
            
            # Forward pass
            verts_pred, delta_verts = model(verts_in, edge_index)
            
            # Loss
            loss_dict = deformation_loss(
                verts_pred=verts_pred,
                verts_target=verts_target,
                verts_in=verts_in,
                edge_index=edge_index,
                faces=faces,
                w_vertex=1.0,
                w_edge=0.1,
                w_laplacian=0.1,
                w_volume=0.01,
            )
            
            loss = loss_dict["total"]
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Print progress
        print(f"Epoch {epoch:3d}/{num_epochs} | Loss: {avg_loss:.6f}")
        with open("loss_log.txt", "a") as f:
            f.write(f"{epoch}, {avg_loss}\n")
        
        # Save checkpoints
        if epoch % 10 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"model_epoch{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f" Checkpoint saved: {ckpt_path}")
    
    print("TRAINING COMPLETE!")
    print(f"Final loss: {avg_loss:.6f}")
    print(f"Checkpoints saved in: {checkpoint_dir}/")


if __name__ == "__main__":
    train(
        dataset_root="data/deformation_dataset/train",
        num_epochs=100,
        lr=1e-3,
        checkpoint_dir="checkpoints_deformation"
    )