import numpy as np

K_MODES = 8
DECAY_POWER = 1.5

def sample_coefficients(n_samples, K, decay_power, rng):
    """
    For each of n_samples functions, draw random amplitudes and phases.
    
    Returns:
        coeffs: shape (n_samples, K) — amplitudes c_k
        phases:  shape (n_samples, K) — phases φ_k
    """

    k = np.arange(1, K + 1)
    decay = 1.0 / (k ** decay_power)

    coeffs = rng.normal(size=(n_samples, K)) * decay
    phases = rng.uniform(0, 2 * np.pi, size=(n_samples, K))
    return coeffs, phases

def evaluate(coeffs, phases, x):
    """
    Evaluate a(x) and u(x) for a batch of functions  on grid points x.

    Args:
        coeffs: (N, K)
        phases:  (N, K)
        x:       (S,) — grid points

    Returns:
        a: (N, S)
        u: (N, S)
    """

    N, K = coeffs.shape
    k = np.arange(1, K + 1).reshape(1, K, 1) # (1, K, 1)
    c = coeffs[:, :, None] # (N, K, 1)
    p = phases[:, :, None] # (N, K, 1)
    xg = x.reshape(1, 1, -1) # (1, 1, S)

    arg = 2 * np.pi * k * xg + p # (N, K, S)
    a = (c * np.sin(arg)).sum(axis=1) # (N, S)
    u = (c * (np.cos(p) - np.cos(arg)) / (2 * np.pi * k)).sum(axis=1) # (N, S)

    return a, u

def make_grid(resolution):
    """Create a grid of points in [0, 1]."""
    return np.linspace(0.0, 1.0, resolution)