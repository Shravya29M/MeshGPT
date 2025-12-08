#!/usr/bin/env python3

import os
import torch
import numpy as np
from tqdm import tqdm

from scripts.pytorch_ds import ModelNet10PC
from scripts.transformer_folding_ae  import TransformerFoldingAE


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def export_reconstructions(
        data_root="data/modelnet10_pc_2048",
        checkpoint="checkpoints_transformer_pp/transformer_foldingpp_epoch99.pth",
        output_dir="exported_pointclouds",
        num_samples=20  # number of test samples to export
    ):

    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    print("Using device:", device)

    # Load TEST dataset only
    test_ds = ModelNet10PC(data_root, split="test", val_ratio=0.1)
    print("Test samples:", len(test_ds))

    # Load model
    model = TransformerFoldingAE(num_points=2048, latent_dim=256).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # Choose random sample indices
    indices = np.random.choice(len(test_ds), size=num_samples, replace=False)

    for idx in tqdm(indices, desc="Exporting point clouds"):
        original = test_ds[idx]  # (2048,3)
        pc_input = torch.tensor(original).unsqueeze(0).to(device)

        # Run reconstruction
        with torch.no_grad():
            recon, _ = model(pc_input)
            recon = recon.squeeze(0).cpu().numpy()

        # Make subfolder by class name
        sample_path = test_ds.files[idx]
        class_name = sample_path.split("/")[-3]     # /class/test/file
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Save original & reconstruction
        base_name = os.path.basename(sample_path).replace(".npy", "")
        np.save(os.path.join(class_dir, f"{base_name}_original.npy"), original)
        np.save(os.path.join(class_dir, f"{base_name}_recon.npy"), recon)

    print(f"\n✨ Export complete! Saved {num_samples} samples to {output_dir}/\n")


if __name__ == "__main__":
    export_reconstructions()
