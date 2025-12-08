import torch
import numpy as np

def edges_from_faces(faces: torch.Tensor, undirected: bool = True) -> torch.Tensor:
    """Build edge_index from triangle faces"""
    device = faces.device
    
    v0 = faces[:, 0]
    v1 = faces[:, 1]
    v2 = faces[:, 2]
    
    e01 = torch.stack([v0, v1], dim=1)
    e12 = torch.stack([v1, v2], dim=1)
    e20 = torch.stack([v2, v0], dim=1)
    
    edges = torch.cat([e01, e12, e20], dim=0)
    
    if undirected:
        edges_rev = torch.stack([edges[:, 1], edges[:, 0]], dim=1)
        edges = torch.cat([edges, edges_rev], dim=0)
    
    mask = edges[:, 0] != edges[:, 1]
    edges = edges[mask]
    
    if edges.device.type == 'cpu':
        edges_np = edges.numpy()
        edges_unique = np.unique(edges_np, axis=0)
        edges = torch.from_numpy(edges_unique).to(device)
    else:
        edges = torch.unique(edges, dim=0)
    
    edges = edges.t().contiguous()
    return edges


def knn_graph(verts: torch.Tensor, k: int = 16, loop: bool = False) -> torch.Tensor:
    """Build kNN edge_index"""
    device = verts.device
    N = verts.shape[0]
    
    with torch.no_grad():
        diff = verts[:, None, :] - verts[None, :, :]
        dist2 = torch.sum(diff * diff, dim=-1)
        
        if not loop:
            idx = torch.arange(N, device=device)
            dist2[idx, idx] = float("inf")
        
        _, nn_idx = torch.topk(dist2, k=k, largest=False)
        
        src = torch.repeat_interleave(torch.arange(N, device=device), k)
        dst = nn_idx.reshape(-1)
        
        edges = torch.stack([src, dst], dim=0)
        
        if not loop:
            mask = edges[0] != edges[1]
            edges = edges[:, mask]
    
    return edges


def build_edges(verts: torch.Tensor, faces: torch.Tensor = None, use_knn_if_no_faces: bool = True, k: int = 16) -> torch.Tensor:
    """Unified edge builder"""
    if faces is not None and faces.numel() > 0:
        return edges_from_faces(faces)
    
    if use_knn_if_no_faces:
        return knn_graph(verts, k=k)
    
    raise ValueError("No faces provided and KNN mode disabled.")