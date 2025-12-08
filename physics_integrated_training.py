import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt

from scripts.pytorch_ds import ModelNet10PC
from scripts.transformer_folding_ae import (
    TransformerFoldingAE,
    chamfer_distance,
    repulsion_loss,
    smoothness_loss
)

try:
    from taichi_mesh import pointcloud_quality_score
    TAICHI_AVAILABLE = True
    print("✓ Taichi point cloud evaluator available")
except ImportError:
    TAICHI_AVAILABLE = False
    print("⚠ Taichi not available")


def get_device():
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_batch_quality_taichi(recon_batch, max_samples=4):
    """
    Evaluate point cloud quality using Taichi (GPU-accelerated).
    NO MESHING REQUIRED!
    
    Args:
        recon_batch: (B, N, 3) torch tensor
        max_samples: max samples to evaluate per batch
    
    Returns:
        dict with quality metrics
    """
    if not TAICHI_AVAILABLE:
        return None
    
    B = min(recon_batch.shape[0], max_samples)
    quality_scores = []
    successful = 0
    
    for i in range(B):
        points_np = recon_batch[i].detach().cpu().numpy()
        
        try:
            # Direct Taichi computation on point cloud
            score = pointcloud_quality_score(points_np, return_detailed=False)
            quality_scores.append(score)
            successful += 1
        except Exception as e:
            print(f"  Quality computation failed: {e}")
            quality_scores.append(100.0)
    
    if successful == 0:
        return None
    
    return {
        'mean_quality': np.mean(quality_scores),
        'std_quality': np.std(quality_scores),
        'min_quality': np.min(quality_scores),
        'max_quality': np.max(quality_scores),
        'success_rate': successful / B,
        'all_scores': quality_scores
    }


def train_with_taichi_pointcloud_metrics(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=80,
    ckpt_dir="checkpoints_taichi_pc",
    quality_eval_freq=2,  # Can evaluate more often since it's faster!
):
    """
    Training with Taichi-based point cloud quality evaluation.
    
    Key advantages:
    - No meshing required (faster)
    - Can evaluate more frequently
    - Direct quality feedback
    - GPU-accelerated computation
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Track metrics
    history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'train_quality': [],
        'val_quality': []
    }
    
    # Adaptive hyperparameters
    repulsion_weight = 0.15
    smoothness_weight = 0.05
    
    # Quality-based early stopping
    best_val_quality = float('inf')
    patience = 10
    patience_counter = 0
    
    print("\n" + "="*70)
    print("TRAINING WITH TAICHI POINT CLOUD QUALITY METRICS")
    print("="*70)
    print(f"Quality evaluation every {quality_eval_freq} epochs")
    print(f"Direct point cloud analysis (NO meshing required)")
    print("="*70 + "\n")
    
    for epoch in range(1, num_epochs + 1):
        # ==================== TRAINING ====================
        model.train()
        train_loss = 0.0
        train_chamfer = 0.0
        
        for pc in train_loader:
            pc = pc.to(device)
            optimizer.zero_grad()
            
            recon, _ = model(pc)
            
            # Standard losses
            loss_ch = chamfer_distance(pc, recon)
            loss_rep = repulsion_loss(recon)
            loss_smooth = smoothness_loss(recon)
            
            # Combined loss (adaptive weights)
            loss = (
                loss_ch +
                repulsion_weight * loss_rep +
                smoothness_weight * loss_smooth
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_chamfer += loss_ch.item()
        
        train_loss /= len(train_loader)
        train_chamfer /= len(train_loader)
        
        # ==================== VALIDATION ====================
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for pc in val_loader:
                pc = pc.to(device)
                recon, _ = model(pc)
                val_loss += chamfer_distance(pc, recon).item()
        
        val_loss /= len(val_loader)
        
        # ==================== QUALITY EVALUATION (TAICHI) ====================
        train_quality_metrics = None
        val_quality_metrics = None
        
        if TAICHI_AVAILABLE and epoch % quality_eval_freq == 0:
            print(f"\n  [Epoch {epoch}] Computing quality with Taichi...")
            
            with torch.no_grad():
                # Training quality
                train_batch = next(iter(train_loader)).to(device)
                train_recon, _ = model(train_batch)
                train_quality_metrics = evaluate_batch_quality_taichi(train_recon)
                
                # Validation quality
                val_batch = next(iter(val_loader)).to(device)
                val_recon, _ = model(val_batch)
                val_quality_metrics = evaluate_batch_quality_taichi(val_recon)
            
            if train_quality_metrics and val_quality_metrics:
                train_q = train_quality_metrics['mean_quality']
                val_q = val_quality_metrics['mean_quality']
                
                # Store history
                history['epochs'].append(epoch)
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_quality'].append(train_q)
                history['val_quality'].append(val_q)
                
                print(f"\n  Quality Metrics (Taichi on Point Clouds):")
                print(f"    Train Quality: {train_q:.4f} ± {train_quality_metrics['std_quality']:.4f}")
                print(f"    Val Quality:   {val_q:.4f} ± {val_quality_metrics['std_quality']:.4f}")
                print(f"    Min/Max:       {val_quality_metrics['min_quality']:.4f} / {val_quality_metrics['max_quality']:.4f}")
                
                # ==================== ADAPTIVE HYPERPARAMETERS ====================
                if len(history['val_quality']) >= 2:
                    prev_q = history['val_quality'][-2]
                    curr_q = history['val_quality'][-1]
                    
                    # Quality degrading
                    if curr_q > prev_q * 1.15:  # 15% worse
                        repulsion_weight = min(repulsion_weight * 1.1, 0.3)
                        smoothness_weight = min(smoothness_weight * 1.1, 0.12)
                        print(f"\n  ⚠ Quality degrading! Increasing regularization:")
                        print(f"    Repulsion:  {repulsion_weight:.4f}")
                        print(f"    Smoothness: {smoothness_weight:.4f}")
                        patience_counter += 1
                    
                    # Quality improving
                    elif curr_q < prev_q * 0.95:  # 5% better
                        repulsion_weight = max(repulsion_weight * 0.98, 0.10)
                        smoothness_weight = max(smoothness_weight * 0.98, 0.03)
                        print(f"\n  ✓ Quality improving!")
                        patience_counter = 0
                    
                    else:
                        patience_counter += 1
                
                # ==================== EARLY STOPPING ====================
                if val_q < best_val_quality:
                    best_val_quality = val_q
                    patience_counter = 0
                    
                    # Save best model
                    best_path = os.path.join(ckpt_dir, "best_model.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'quality': val_q,
                    }, best_path)
                    print(f"  ★ New best quality! Saved to {best_path}")
                
                if patience_counter >= patience:
                    print(f"\n  Early stopping: No improvement for {patience} evaluations")
                    break
        
        # ==================== LOGGING ====================
        log_str = f"Epoch {epoch}/{num_epochs} | Loss: {train_loss:.6f} | Val: {val_loss:.6f}"
        
        if train_quality_metrics:
            log_str += f" | Quality: {train_quality_metrics['mean_quality']:.4f}"
        
        print(log_str)
        
        # ==================== CHECKPOINTING ====================
        if epoch % 10 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history,
                'repulsion_weight': repulsion_weight,
                'smoothness_weight': smoothness_weight
            }, ckpt_path)
            print(f"  → Checkpoint: {ckpt_path}")
    
    # ==================== FINAL PLOTS ====================
    if len(history['epochs']) > 0:
        plot_training_with_quality(history, save_path=os.path.join(ckpt_dir, 'training_curves.png'))
    
    return history


def plot_training_with_quality(history, save_path='training_curves.png'):
    """Plot training curves with quality metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    all_epochs = list(range(1, len(history['train_loss']) + 1))
    quality_epochs = history['epochs']
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    ax1.plot(all_epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax1.plot(all_epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Chamfer Distance')
    ax1.set_title('Reconstruction Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Point Cloud Quality (Taichi)
    ax2 = axes[0, 1]
    if history['train_quality']:
        ax2.plot(quality_epochs, history['train_quality'], 'b-o', 
                label='Train Quality', linewidth=2, markersize=6)
        ax2.plot(quality_epochs, history['val_quality'], 'r-o',
                label='Val Quality', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Quality Score (lower = better)')
    ax2.set_title('Point Cloud Quality (Taichi)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss vs Quality Correlation
    ax3 = axes[1, 0]
    if history['train_quality']:
        # Match up losses with quality evaluations
        quality_epoch_losses = [history['train_loss'][e-1] for e in quality_epochs if e <= len(history['train_loss'])]
        sc = ax3.scatter(quality_epoch_losses, history['train_quality'][:len(quality_epoch_losses)],
                        s=100, alpha=0.6, c=quality_epochs[:len(quality_epoch_losses)], cmap='viridis')
        ax3.set_xlabel('Reconstruction Loss')
        ax3.set_ylabel('Quality Score')
        ax3.set_title('Loss vs Quality Trade-off')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax3, label='Epoch')
    
    # Plot 4: Quality Improvement Rate
    ax4 = axes[1, 1]
    if len(history['val_quality']) > 1:
        improvements = []
        for i in range(1, len(history['val_quality'])):
            prev = history['val_quality'][i-1]
            curr = history['val_quality'][i]
            improvement = ((prev - curr) / prev) * 100
            improvements.append(improvement)
        
        colors = ['g' if x > 0 else 'r' for x in improvements]
        ax4.bar(quality_epochs[1:], improvements, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='k', linestyle='--', linewidth=1)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Quality Improvement (%)')
        ax4.set_title('Quality Change Per Evaluation')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training curves saved to {save_path}")
    plt.close()


def main():
    data_root = "data/modelnet10_pc_2048"
    device = get_device()
    
    print("\n" + "="*70)
    print("TAICHI POINT CLOUD QUALITY TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Taichi: {TAICHI_AVAILABLE}")
    print("="*70 + "\n")
    
    # Load data
    train_ds = ModelNet10PC(data_root, split="train")
    val_ds = ModelNet10PC(data_root, split="val")
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    
    # Model
    model = TransformerFoldingAE(num_points=2048, latent_dim=256).to(device)
    
    # Load pretrained
    #load_ckpt = "checkpoints_foldingnet/foldingnet_epoch200.pth"
    load_ckpt=""
    if os.path.exists(load_ckpt):
        checkpoint = torch.load(load_ckpt, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Loaded: {load_ckpt}\n")
    
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    # Train
    history = train_with_taichi_pointcloud_metrics(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=280,
        ckpt_dir="checkpoints_taichi_pc",
        quality_eval_freq=2  # Every 2 epochs (faster than meshing!)
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    if history['val_quality']:
        best_idx = np.argmin(history['val_quality'])
        best_epoch = history['epochs'][best_idx]
        best_quality = history['val_quality'][best_idx]
        print(f"Best quality: {best_quality:.4f} at epoch {best_epoch}")
        print(f"Final quality: {history['val_quality'][-1]:.4f}")
        improvement = ((history['val_quality'][0] - history['val_quality'][-1]) / history['val_quality'][0]) * 100
        print(f"Improvement: {improvement:.1f}%")


if __name__ == "__main__":
    main()