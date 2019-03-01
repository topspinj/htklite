import numpy as np

def get_principal_components(m):
    """Takes a matrix m and returns the 3x3 matrix of
    principal components. Computed using Singular Value 
    Decomposition (SVD).
    """
    return np.linalg.svd(m.astype(float), full_matrices=False)[0].astype(m.dtype)


def magnitude(m):
    """Get the magnitude of each column vector in a matrix"""
    return np.sqrt((m ** 2).sum(0))


def normalize(m):
    """Normalize each column vector in a matrix"""
    return m / magnitude(m)
