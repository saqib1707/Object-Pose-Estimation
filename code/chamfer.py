import torch
import numpy as np

# helper functions for computing Chamfer distance
def bpdist2(feature1, feature2, data_format='NWC'):
    """This version has a high memory usage but more compatible(accurate) with optimized Chamfer Distance."""
    if data_format == 'NCW':
        diff = feature1.unsqueeze(3) - feature2.unsqueeze(2)
        distance = torch.sum(diff ** 2, dim=1)
    elif data_format == 'NWC':
        diff = feature1.unsqueeze(2) - feature2.unsqueeze(1)
        distance = torch.sum(diff ** 2, dim=3)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))
    return distance

# params:
#   xyz1: 1 x points x 3, point cloud 1 (pc1)
#   xyz2: 1 x points x 3, point cloud 2 (pc2)
# output:
#   dist1: 1 x points in point cloud 1, the nearest neighbor distance for each point of pc1 to pc2
#   idx1: 1 x points in point cloud 1, for each point of pc1, index of the nearest neighbor in pc2
#   dist2: 1 x points in point cloud 2, the nearest neighbor distance for each point of pc2 to pc1
#   idx2: 1 x points in point cloud 2, for each point of pc2, index of the nearest neighbor in pc1
def Chamfer_distance_torch(xyz1, xyz2, data_format='NWC'):
    assert torch.is_tensor(xyz1) and xyz1.dim() == 3
    assert torch.is_tensor(xyz2) and xyz2.dim() == 3
    if data_format == 'NCW':
        assert xyz1.size(1) == 3 and xyz2.size(1) == 3
    elif data_format == 'NWC':
        assert xyz1.size(2) == 3 and xyz2.size(2) == 3
    distance = bpdist2(xyz1, xyz2, data_format)
    dist1, idx1 = distance.min(2)
    dist2, idx2 = distance.min(1)
    return dist1, idx1, dist2, idx2
