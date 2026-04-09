import torch
import random

def deform_upward(verts, amount=0.3):
    """Simple upward translation"""
    v = verts.clone()
    v[:, 2] += amount
    return v


def deform_bump(verts, center=(0,0,0.5), radius=0.6, strength=0.3):
    """Local bump deformation"""
    v = verts.clone()
    cx, cy, cz = center
    dist = torch.sqrt((v[:,0]-cx)**2 + (v[:,1]-cy)**2 + (v[:,2]-cz)**2)
    mask = dist < radius
    v[mask, 2] += strength * (1 - dist[mask] / radius)
    return v


def deform_twist(verts, factor=1.0):
    """Twist deformation"""
    v = verts.clone()
    theta = factor * v[:,2]
    x = v[:,0]
    y = v[:,1]
    v[:,0] = x * torch.cos(theta) - y * torch.sin(theta)
    v[:,1] = x * torch.sin(theta) + y * torch.cos(theta)
    return v


def deform_sitting(verts, compression=0.15):
    """Simulate someone sitting - compress the seat area"""
    v = verts.clone()
    seat_mask = (v[:, 2] > -0.3) & (v[:, 2] < 0.1)
    v[seat_mask, 2] -= compression
    return v


def deform_sag(verts, amount=0.1):
    """Gentle downward sag from weight/age"""
    v = verts.clone()
    center_dist = torch.sqrt(v[:, 0]**2 + v[:, 1]**2)
    sag_factor = torch.exp(-center_dist * 2)
    v[:, 2] -= amount * sag_factor
    return v


def deform_squeeze(verts, factor=0.1):
    """Compress horizontally (squeeze from sides)"""
    v = verts.clone()
    v[:, 0] *= (1 - factor)
    v[:, 1] *= (1 - factor)
    return v


def deform_bend(verts, axis='x', angle=0.3):
    """Bend the object along an axis"""
    v = verts.clone()
    
    z_min, z_max = v[:, 2].min(), v[:, 2].max()
    if z_max - z_min < 1e-8:
        return v
    
    z_normalized = (v[:, 2] - z_min) / (z_max - z_min)
    
    if axis == 'x':
        v[:, 1] += angle * z_normalized * torch.abs(v[:, 0])
    elif axis == 'y':
        v[:, 0] += angle * z_normalized * torch.abs(v[:, 1])
    
    return v


def deform_taper(verts, factor=0.3, direction='top'):
    """Make top/bottom narrower"""
    v = verts.clone()
    
    z_min, z_max = v[:, 2].min(), v[:, 2].max()
    if z_max - z_min < 1e-8:
        return v
    
    z_normalized = (v[:, 2] - z_min) / (z_max - z_min)
    
    if direction == 'top':
        scale = 1.0 - factor * z_normalized
    else:
        scale = 1.0 - factor * (1.0 - z_normalized)
    
    v[:, 0] *= scale
    v[:, 1] *= scale
    
    return v


def deform_shear(verts, amount=0.2, axis='x'):
    """Shear/slant the object"""
    v = verts.clone()
    
    z_min, z_max = v[:, 2].min(), v[:, 2].max()
    if z_max - z_min < 1e-8:
        return v
    
    z_normalized = (v[:, 2] - z_min) / (z_max - z_min)
    
    if axis == 'x':
        v[:, 0] += amount * z_normalized
    else:
        v[:, 1] += amount * z_normalized
    
    return v


def deform_bulge(verts, strength=0.2):
    """Make the middle bulge out"""
    v = verts.clone()
    
    z_min, z_max = v[:, 2].min(), v[:, 2].max()
    if z_max - z_min < 1e-8:
        return v
    
    z_normalized = (v[:, 2] - z_min) / (z_max - z_min)
    bulge_factor = torch.sin(z_normalized * 3.14159)
    
    scale = 1.0 + strength * bulge_factor
    v[:, 0] *= scale
    v[:, 1] *= scale
    
    return v


def deform_stretch(verts, axis='z', factor=0.3):
    """Stretch along an axis"""
    v = verts.clone()
    
    if axis == 'z':
        v[:, 2] *= (1 + factor)
    elif axis == 'x':
        v[:, 0] *= (1 + factor)
    else:
        v[:, 1] *= (1 + factor)
    
    return v



def deform_ripple(verts, frequency=5.0, amplitude=0.1):
    """Add wave-like ripples"""
    v = verts.clone()
    
    dist = torch.sqrt(v[:, 0]**2 + v[:, 1]**2)
    wave = torch.sin(dist * frequency) * amplitude
    v[:, 2] += wave
    
    return v


def deform_dent(verts, num_dents=3, strength=0.15):
    """Add multiple random dents"""
    v = verts.clone()
    
    for _ in range(num_dents):
        center_x = random.uniform(-0.5, 0.5)
        center_y = random.uniform(-0.5, 0.5)
        center_z = random.uniform(-0.5, 0.5)
        
        dist = torch.sqrt(
            (v[:, 0] - center_x)**2 + 
            (v[:, 1] - center_y)**2 + 
            (v[:, 2] - center_z)**2
        )
        
        dent = strength * torch.exp(-dist * 10)
        
        direction = v - torch.tensor([center_x, center_y, center_z])
        direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-8)
        v -= direction * dent.unsqueeze(1)
    
    return v


def deform_inflate(verts, amount=0.3):
    """Inflate like a balloon"""
    v = verts.clone()
    
    center = v.mean(dim=0)
    direction = v - center
    distance = torch.norm(direction, dim=1, keepdim=True)
    
    direction = direction / (distance + 1e-8)
    v += direction * amount
    
    return v


