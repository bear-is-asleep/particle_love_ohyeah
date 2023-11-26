import numpy as np
import numba as nb
import torch

@nb.jit(nopython=True)
def get_displacement_cache(positions,grid_masks):
    """
    Calculate the displacement vectors between all pairs of positions in a list of positions.

    Parameters:
    positions (numpy.ndarray): An array of positions to calculate the displacement vectors PxD

    Returns:
    r_dists(numpy.ndarray): Matrix of displacements PxPxD
    """
    if grid_masks is None:
        r_dists = np.zeros((positions.shape[0],positions.shape[0],positions.shape[1]))
        for i, pos in enumerate(positions):
            for j, other_pos in enumerate(positions):
                if i == j: continue
                r_dists[i,j] = other_pos - pos
        return r_dists


nb.jit(nopython=True)
def get_separation_cache(radii):
    """
    Calculate the separation distances between all pairs of radii in a list of radii.

    Parameters:
    radii (numpy.ndarray): An array of radii to calculate the separation distances Px1

    Returns:
    r_dists(numpy.ndarray): Matrix of separation distances PxP
    """
    r_dists = np.zeros((radii.shape[0],radii.shape[0]))
    for i, r in enumerate(radii):
        for j, other_r in enumerate(radii):
            if i == j: continue
            r_dists[i,j] = r + other_r
    return r_dists