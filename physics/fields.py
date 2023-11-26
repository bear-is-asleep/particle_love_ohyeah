import torch
from . import interactions

def trilinear_interpolation(grid, points, device):
    """
    Performs trilinear interpolation on a 3D grid given points.
    :param grid: a 3D grid of shape (D, H, W) with values at the vertices
    """
    D, H, W = grid.shape
  
    # Convert points to be in grid coordinates
    points = points * torch.tensor([D - 1, H - 1, W - 1], device=points.device, dtype=points.dtype)

    # Get the integer and fractional parts of the points
    points_floor = torch.floor(points).long()
    points_frac = points - points_floor.float()

    # Identify the vertices of the cubes surrounding the points
    x0 = points_floor[:, 0]
    x1 = x0 + 1
    y0 = points_floor[:, 1]
    y1 = y0 + 1
    z0 = points_floor[:, 2]
    z1 = z0 + 1

    # Ensure the indices are within the grid boundaries
    if x0 < 0 or x0 >= D - 1 or y0 < 0 or y0 >= H - 1 or z0 < 0 or z0 >= W - 1:
        return torch.zeros(points.shape[0], device=device, dtype=grid.dtype)

    # Ensure the indices are within the grid boundaries
    x0 = torch.clamp(x0, 0, D - 1)
    x1 = torch.clamp(x1, 0, D - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    y1 = torch.clamp(y1, 0, H - 1)
    z0 = torch.clamp(z0, 0, W - 1)
    z1 = torch.clamp(z1, 0, W - 1)

    # Gather the values at the vertices
    v000 = grid[x0, y0, z0]
    v001 = grid[x0, y0, z1]
    v010 = grid[x0, y1, z0]
    v011 = grid[x0, y1, z1]
    v100 = grid[x1, y0, z0]
    v101 = grid[x1, y0, z1]
    v110 = grid[x1, y1, z0]
    v111 = grid[x1, y1, z1]

    # Interpolate along the x-axis
    vx00 = v000 * (1 - points_frac[:, 0]) + v100 * points_frac[:, 0]
    vx01 = v001 * (1 - points_frac[:, 0]) + v101 * points_frac[:, 0]
    vx10 = v010 * (1 - points_frac[:, 0]) + v110 * points_frac[:, 0]
    vx11 = v011 * (1 - points_frac[:, 0]) + v111 * points_frac[:, 0]

    # Interpolate along the y-axis
    vxy0 = vx00 * (1 - points_frac[:, 1]) + vx10 * points_frac[:, 1]
    vxy1 = vx01 * (1 - points_frac[:, 1]) + vx11 * points_frac[:, 1]

    # Interpolate along the z-axis
    vxyz = vxy0 * (1 - points_frac[:, 2]) + vxy1 * points_frac[:, 2]

    return vxyz

def compute_gravity_field(distances,separations,masses,edge_mask,G=1):
    pass