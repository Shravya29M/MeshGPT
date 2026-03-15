import torch

def deformation_loss(
    verts_pred,
    verts_target,
    verts_in,
    edge_index,
    faces=None,
    w_vertex=1.0,
    w_edge=0.1,
    w_laplacian=0.1,
    w_volume=0.01,
):
    
    # Vertex position loss
    loss_vertex = torch.mean((verts_pred - verts_target) ** 2)
    
    # Edge length preservation
    i, j = edge_index[0], edge_index[1]
    edge_len_original = torch.norm(verts_in[i] - verts_in[j], dim=1)
    edge_len_pred = torch.norm(verts_pred[i] - verts_pred[j], dim=1)
    loss_edge = torch.mean((edge_len_pred - edge_len_original) ** 2)
    
    # Laplacian smoothness
    laplacian_original = verts_in[i] - verts_in[j]
    laplacian_pred = verts_pred[i] - verts_pred[j]
    loss_laplacian = torch.mean((laplacian_pred - laplacian_original) ** 2)
    
    # Volume preservation (simplified)
    if faces is not None and faces.numel() > 0:
        try:
            v0 = verts_pred[faces[:, 0]]
            v1 = verts_pred[faces[:, 1]]
            v2 = verts_pred[faces[:, 2]]
            volumes = torch.abs(torch.sum(v0 * torch.cross(v1, v2, dim=1), dim=1))
            loss_volume = torch.var(volumes)
        except:
            loss_volume = torch.tensor(0.0, device=verts_pred.device)
    else:
        loss_volume = torch.tensor(0.0, device=verts_pred.device)
    
    # Total loss
    total_loss = (
        w_vertex * loss_vertex +
        w_edge * loss_edge +
        w_laplacian * loss_laplacian +
        w_volume * loss_volume
    )
    
    return {
        "total": total_loss,
        "vertex": loss_vertex,
        "edge": loss_edge,
        "laplacian": loss_laplacian,
        "volume": loss_volume
    }


