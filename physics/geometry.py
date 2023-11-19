import numpy as np
import numba as nb

@nb.jit(nopython=True)
def get_displacement_cache(positions):
    """
    Calculate the displacement vectors between all pairs of positions in a list of positions.

    Parameters:
    positions (numpy.ndarray): An array of positions to calculate the displacement vectors PxD

    Returns:
    r_dists(numpy.ndarray): Matrix of displacements PxPxD
    """
    r_dists = np.zeros((positions.shape[0],positions.shape[0],positions.shape[1]))
    for i, pos in enumerate(positions):
        for j, other_pos in enumerate(positions):
            if i == j: continue
            r_dists[i,j] = other_pos - pos
    return r_dists
