import torch
import torch.nn as nn

class EdgeConvLayer(nn.Module):
    """A single EdgeConv layer"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        assert edge_index.max() < N, f"Edge index {edge_index.max()} >= num nodes {N}"
        
        i = edge_index[0]
        j = edge_index[1]
        
        x_i = x[i]
        x_j = x[j]
        
        edge_feat = torch.cat([x_i, x_j - x_i], dim=-1)
        messages = self.mlp(edge_feat)
        
        C_out = messages.shape[1]
        out = x.new_zeros(N, C_out)
        out.index_add_(0, j, messages)
        
        deg = torch.bincount(j, minlength=N).clamp(min=1).float().view(-1, 1)
        out = out / deg
        
        return out


class EdgeConvDeformationNet(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 3,
        out_dim: int = 3,
    ):
        super().__init__()
        
        layers = [EdgeConvLayer(in_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.append(EdgeConvLayer(hidden_dim, hidden_dim))
        
        self.convs = nn.ModuleList(layers)
        
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, verts: torch.Tensor, edge_index: torch.Tensor):
        x = verts
        
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            if i > 0:
                x = x + x_new
            else:
                x = x_new
        
        delta_verts = self.mlp_out(x)
        verts_pred = verts + delta_verts
        
        return verts_pred, delta_verts