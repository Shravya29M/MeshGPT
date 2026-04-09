import torch
from torch.utils.data import Dataset, DataLoader
import os

class DeformationDataset(Dataset):
    
    def __init__(self, root_dir):
      
        self.root_dir = root_dir
        self.samples = sorted([
            f for f in os.listdir(root_dir) 
            if f.endswith('.pt')
        ])
        
        if len(self.samples) == 0:
            raise ValueError(f"No .pt files found in {root_dir}")
        
        print(f"Loaded {len(self.samples)} samples from {root_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Return (verts_in, faces, verts_target)"""
        sample_path = os.path.join(self.root_dir, self.samples[idx])
        data = torch.load(sample_path)
        
        return (
            data["verts_in"],
            data["faces"],
            data["verts_target"]
        )


def get_dataloader(dataset_root, batch_size=1, shuffle=True):
    dataset = DeformationDataset(dataset_root)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=lambda x: x[0]  # Return single sample since meshes vary in size
    )


def load_single_sample(sample_path):
    """Load a single sample for visualization"""
    data = torch.load(sample_path)
    return (
        data["verts_in"],
        data["faces"],
        data["verts_target"]
    )