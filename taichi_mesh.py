import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)
MAX_POINTS = 10000

num_points = ti.field(dtype=ti.i32, shape=())
points = ti.field(dtype=ti.f32, shape=(MAX_POINTS, 3))

K_NEIGHBORS = 16
knn_indices = ti.field(dtype=ti.i32, shape=(MAX_POINTS, 16))
knn_distances = ti.field(dtype=ti.f32, shape=(MAX_POINTS, 16))

neighbor_variance_sum = ti.field(dtype=ti.f32, shape=())
aspect_ratio_sum = ti.field(dtype=ti.f32, shape=())
density_variance = ti.field(dtype=ti.f32, shape=())
local_uniformity_sum = ti.field(dtype=ti.f32, shape=())

point_neighbor_var = ti.field(dtype=ti.f32, shape=MAX_POINTS)
point_aspect_ratio = ti.field(dtype=ti.f32, shape=MAX_POINTS)
point_density = ti.field(dtype=ti.f32, shape=MAX_POINTS)


@ti.func
def distance_3d(p1_idx: ti.i32, p2_idx: ti.i32) -> ti.f32:
    dx=points[p1_idx, 0]-points[p2_idx, 0]
    dy=points[p1_idx, 1]-points[p2_idx, 1]
    dz=points[p1_idx, 2]-points[p2_idx, 2]
    return ti.sqrt(dx * dx + dy * dy + dz * dz)


@ti.kernel
def reset_accumulators():
    
    neighbor_variance_sum[None]= 0.0
    aspect_ratio_sum[None]=0.0
    density_variance[None]=0.0
    local_uniformity_sum[None]=0.0


@ti.kernel
def upload_points(pts: ti.types.ndarray()):
    num_points[None]=pts.shape[0]
    for i in range(pts.shape[0]):
        points[i, 0]=pts[i, 0]
        points[i, 1]=pts[i, 1]
        points[i, 2]=pts[i, 2]


@ti.kernel
def compute_knn_bruteforce():
    
    N = num_points[None]
    K = 16 
    for i in range(N):
        for k in range(K):
            knn_distances[i, k] = 1e10
            knn_indices[i, k] = -1
        
        for j in range(N):
            if(i==j):
                continue
            dist = distance_3d(i, j)
            if(dist < knn_distances[i, K - 1]):
                pos=K - 1
                for k in range(K - 1):
                    if(dist < knn_distances[i, k]):
                        pos=k
                        break
                k = K - 1
                while k > pos:
                    knn_distances[i, k] = knn_distances[i, k - 1]
                    knn_indices[i, k] = knn_indices[i, k - 1]
                    k -= 1
                knn_distances[i, pos] = dist
                knn_indices[i, pos] = j
@ti.kernel
def compute_point_metrics():
    N = num_points[None]
    K = 16 
    for i in range(N):
        mean_dist = 0.0
        for k in range(K):
            mean_dist += knn_distances[i, k]
        mean_dist /= K
        variance = 0.0
        for k in range(K):
            diff = knn_distances[i, k] - mean_dist
            variance += diff * diff
        variance /= K
        
        point_neighbor_var[i] = variance
        neighbor_variance_sum[None] += variance
        min_dist = 1e10
        max_dist = -1e10
        for k in range(K):
            d = knn_distances[i, k]
            if d < min_dist:
                min_dist = d
            if d > max_dist:
                max_dist = d
        
        aspect = max_dist / (min_dist + 1e-8)
        point_aspect_ratio[i] = aspect
        aspect_ratio_sum[None] += aspect
        point_density[i] = 1.0 / (mean_dist + 1e-8)
        std_dev = ti.sqrt(variance)
        cv = std_dev / (mean_dist + 1e-8)
        local_uniformity_sum[None] += cv


@ti.kernel
def compute_density_variance():  #low variance is better
    N = num_points[None]
    
    mean_density = 0.0
    for i in range(N):
        mean_density+=point_density[i]
    mean_density/=N
    var = 0.0
    for i in range(N):
        diff=point_density[i]-mean_density
        var+=diff * diff
    var /= N
    density_variance[None] = var


def pointcloud_quality_score(points_np, return_detailed=False):
    points_np = np.asarray(points_np, dtype=np.float32)
    if points_np.ndim != 2 or points_np.shape[1] != 3:
        raise ValueError(f"points_np must be (N, 3), got {points_np.shape}")
    N = points_np.shape[0]
    
    if(N > MAX_POINTS):
        raise ValueError(f"Point cloud has {N} points, exceeds MAX_POINTS={MAX_POINTS}")
    if(N < K_NEIGHBORS+1):
        raise ValueError(f"Need at least {K_NEIGHBORS + 1} points, got {N}")
    
    reset_accumulators()
    upload_points(points_np)
    
    compute_knn_bruteforce()
    
    compute_point_metrics()
    compute_density_variance()
    N_actual = num_points[None]
    
    avg_neighbor_var = neighbor_variance_sum[None]/N_actual
    avg_aspect_ratio = aspect_ratio_sum[None]/ N_actual
    density_var = density_variance[None]
    avg_uniformity = local_uniformity_sum[None]/N_actual
    #Combined quality score (lower = better)
    quality_score = (
        avg_neighbor_var * 100.0 +      #Neighbor uniformity (most important)
        (avg_aspect_ratio - 1.0) * 10.0 +  #Aspect ratios (want ~1.0)
        density_var * 50.0 +               #Density distribution
        avg_uniformity * 5.0               #Local uniformity
    )
    
    if not return_detailed:
        return float(quality_score)
    
    #Detailed metrics
    detailed = {
        'quality_score': float(quality_score),
        'avg_neighbor_variance': float(avg_neighbor_var),
        'avg_aspect_ratio': float(avg_aspect_ratio),
        'density_variance': float(density_var),
        'avg_local_uniformity': float(avg_uniformity),
        'num_points': N_actual,
        'k_neighbors': 16  # Hardcoded constant
    }
    
    return quality_score, detailed


def get_per_point_metrics(points_np):
    points_np = np.asarray(points_np, dtype=np.float32)
    
    if points_np.ndim != 2 or points_np.shape[1] != 3:
        raise ValueError(f"points_np must be (N, 3), got {points_np.shape}")
    
    N = points_np.shape[0]
    K = 16
    reset_accumulators()
    upload_points(points_np)
    compute_knn_bruteforce()
    compute_point_metrics()
    neighbor_vars = np.array([point_neighbor_var[i] for i in range(N)])
    aspect_ratios = np.array([point_aspect_ratio[i] for i in range(N)])
    densities = np.array([point_density[i] for i in range(N)])

    knn_idx = np.zeros((N, K), dtype=np.int32)
    knn_dist = np.zeros((N, K), dtype=np.float32)
    for i in range(N):
        for k in range(K):
            knn_idx[i, k] = knn_indices[i, k]
            knn_dist[i, k] = knn_distances[i, k]
    
    return {
        'neighbor_variance': neighbor_vars,
        'aspect_ratio': aspect_ratios,
        'density': densities,
        'knn_indices': knn_idx,
        'knn_distances': knn_dist
    }


def compare_pointclouds(pc1, pc2):
    score1, details1=pointcloud_quality_score(pc1, return_detailed=True)
    score2, details2=pointcloud_quality_score(pc2, return_detailed=True)
    improvement=((score1-score2)/ score1) * 100 if score1 > 0 else 0
    return {
        'pc1_score': score1,
        'pc2_score': score2,
        'improvement_percent': improvement,
        'winner': 'pc1' if score1 < score2 else 'pc2',
        'pc1_details': details1,
        'pc2_details': details2
    }

