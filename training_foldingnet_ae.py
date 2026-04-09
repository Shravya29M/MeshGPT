import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from scripts.pytorch_ds import ModelNet10PC
from scripts.transformer_folding_ae import TransformerFoldingAE, chamfer_distance, repulsion_loss, smoothness_loss


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    data_root = "data/modelnet10_pc_2048"
    device = get_device()
    print("Using device:", device)

    # 70 / 10 / 20 split (already in ModelNet10PC)
    train_ds = ModelNet10PC(data_root, split="train")
    val_ds   = ModelNet10PC(data_root, split="val")

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False)

    model = TransformerFoldingAE(num_points=2048, latent_dim=256).to(device)

    load_ckpt = "checkpoints_foldingnet/foldingnet_epoch200.pth"
    if os.path.exists(load_ckpt):
        model.load_state_dict(torch.load(load_ckpt, map_location=device))
        print(f"Loaded pretrained weights from {load_ckpt}")
    else:
        print("No checkpoint found — training from scratch.")
    optimizer = Adam(model.parameters(), lr=1e-4)

    num_epochs = 80
    ckpt_dir = "checkpoints_foldingnet"
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for pc in train_loader:
            pc = pc.to(device)
            optimizer.zero_grad()

            recon, _ = model(pc)
            loss_ch = chamfer_distance(pc, recon)
            loss_rep = repulsion_loss(recon)          # spreads points
            loss_smooth = smoothness_loss(recon)      # stabilizes neighbors

            loss = (
                loss_ch 
                + 0.15 * loss_rep        # good balance for 2048 pts
                + 0.05 * loss_smooth
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pc in val_loader:
                pc = pc.to(device)
                recon, _ = model(pc)
                val_loss += chamfer_distance(pc, recon).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}/{num_epochs} | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    ckpt_path = os.path.join(ckpt_dir, f"foldingnet_epoch{epoch}.pth")
    torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    main()
