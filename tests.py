"""
The design of the code for the permutation test is based on: https://github.com/ikjunchoi/rff-mmd from https://arxiv.org/pdf/2407.08976.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import decide, independent_permutation

from mmd import MMD2b, MMD2b_from_K

from kernels import RBFkernel

from nystrom import nys_inds, nystrom_features

import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# from nystrom_split import NystromApproximator


# ----------------------------
# Permutation test
# ----------------------------
def MMDbtest(Z: np.ndarray,
              bw: float,
              seed: int = None,
              alpha: float = 0.05,
              B: int = 199,
              plot: bool = False):
    """
    Perform a biased MMD permutation test with kernel precomputation.

    Args:
        Z (np.ndarray): Data matrix of shape (2n, d), with both samples stacked.
                        First n rows are group X, last n rows are group Y.
        bw (float): Bandwidth (sigma) parameter for the RBF kernel.
        seed (int, optional): Random seed for reproducibility. Default: None.
        alpha (float): Test significance level. Default: 0.05.
        B (int): Number of random permutations. Default: 199.
        plot (bool): If True, plot permutation distribution. Default: False.

    Returns:
        output (int): 1 if null hypothesis rejected, 0 otherwise.
        dt (float): Computation time in seconds.
        ntot (int): Total number of samples (2n).
    """
    start = time.time()
    rng = np.random.default_rng(seed)

    ntot, d = Z.shape
    n = ntot // 2

    # --- Change #1: precompute kernel once ---
    K = RBFkernel(Z, Z, bw)

    # Observed statistic: split first n vs last n
    A_obs = np.arange(n)
    t_obs = MMD2b_from_K(K, A_obs)

    # Permutation distribution
    H0 = np.empty(B + 1, dtype=float)
    for i in tqdm(range(B)):
        perm = rng.permutation(ntot)   # permuted indices
        A = perm[:n]
        H0[i] = MMD2b_from_K(K, A)

    # Add observed statistic
    H0[-1] = t_obs

    # Decision rule (your existing decide() function is used here)
    output, thr = decide(H0, t_obs, B, alpha)

    if plot:
        values, bins, patches = plt.hist(H0, label=r"$H_0$")
        plt.vlines(thr, 0, max(values) * 0.8, color='red', label=f"{alpha}-level threshold")
        plt.vlines(t_obs, 0, max(values) * 0.8, color='black', label="Observed test statistic")
        plt.legend()
        plt.show()

    dt = time.time() - start
    return output, dt, ntot


# def MMDb_test(Z, bw, seed=None, alpha=0.05, B=199, plot=False):
#     """
#     Performs the MMD (Maximum Mean Discrepancy) permutation test.
    
#     Parameters:
#         Z (array): Feature matrix of shape (2n, d).
#         bw (float): Bandwidth parameter for the kernel.
#         seed (int, optional): Random seed for reproducibility.
#         alpha (float): Significance level for the test.
#         B (int): Number of permutations.
#         plot (bool): Whether to plot the distribution of permuted statistics.

#     Returns:
#         output (int): 1 if null hypothesis is rejected, 0 otherwise.
#         dt (float): Computation time in seconds.
#         ntot (int): Total number of samples.
#     """
#     start = time.time()  # Start computation timer
#     rng = np.random.default_rng(seed)  # Random number generator

#     ntot, d = Z.shape  # Total number of samples and feature dimension
#     n = int(ntot / 2)  # Sample size per group

#     # Compute observed test statistic
#     t_obs = MMD2b(Z[:n], Z[n:], bw)

#     # Generate permuted test statistics
#     H0 = np.empty(B + 1)
#     for i in tqdm(range(B)):
#         Z_perm = rng.permutation(Z)  # Permute data
#         H0[i] = MMD2b(Z_perm[:n], Z_perm[n:], bw)
    
#     # Add observed test statistic to permutation distribution
#     H0[-1] = t_obs

#     # Compute test decision and threshold
#     output, thr = decide(H0, t_obs, B, alpha)

#     # Plot histogram if requested
#     if plot:
#         values, bins, patches = plt.hist(H0, label=r"$H_0$")
#         plt.vlines(thr, 0, max(values) * 0.8, color='red', label=f"{alpha}-level threshold")
#         plt.vlines(t_obs, 0, max(values) * 0.8, color='black', label="Observed test statistic")
#         plt.legend()
#         plt.show()

#     dt = time.time() - start  # Compute total execution time
#     return output, dt, ntot

def rMMDtest(Z, seed=None, bandwidth=1, alpha=0.05, kernel='gaussian', R=20, B=199):
    """
    Performs the Random Fourier Feature approximation MMD test.
    
    Parameters:
        Z (array): Feature matrix of shape (2n, d).
        seed (int, optional): Random seed for reproducibility.
        bandwidth (float): Kernel bandwidth parameter.
        alpha (float): Significance level for the test.
        kernel (str): Kernel type ('gaussian' supported).
        R (int): Number of random Fourier features.
        B (int): Number of permutations.

    Returns:
        output (int): 1 if null hypothesis is rejected, 0 otherwise.
        dt (float): Computation time in seconds.
        R (int): Number of random Fourier features used.
    """
    start = time.time()
    rng = np.random.default_rng(seed)  # Random number generator

    ntot, d = Z.shape  # Total samples and feature dimension
    n = m = int(ntot / 2)  # Sample size per group

    # Generate random Fourier features
    if kernel == 'gaussian':
        omegas = np.sqrt(2) / bandwidth * rng.normal(size=(R, d))
    else:
        raise ValueError("Currently only 'gaussian' kernel is supported.")

    omegas_Z = np.dot(Z, omegas.T)
    cos_feature = (1 / np.sqrt(R)) * np.cos(omegas_Z)
    sin_feature = (1 / np.sqrt(R)) * np.sin(omegas_Z)
    psi_Z = np.concatenate((cos_feature, sin_feature), axis=1)

    # Perform permutation test
    I_1 = np.concatenate((np.ones(m), np.zeros(n)))
    I = np.tile(I_1, (B + 1, 1))
    I_X = independent_permutation(I, rng, axis=1)
    I_X[0] = I_1  # Keep original order for the first case
    I_Y = 1 - I_X

    # Compute test statistics
    bar_Z_B_piX = (1 / m) * I_X @ psi_Z
    bar_Z_B_piY = (1 / n) * I_Y @ psi_Z
    T = bar_Z_B_piX - bar_Z_B_piY
    V = np.sum(T ** 2, axis=1)

    rMMD2 = V[0]
    output, _ = decide(V, rMMD2, B, alpha)

    dt = time.time() - start  # Compute execution time
    return output, dt, R

def NysMMDtest(Z, seed=None, bandwidth=1, alpha=0.05, method='uniform', k=20, B=199):
    """
    Performs the Nyström approximation MMD test.
    
    Parameters:
        Z (array): Feature matrix of shape (2n, d).
        seed (int, optional): Random seed for reproducibility.
        bandwidth (float): Kernel bandwidth parameter.
        alpha (float): Significance level for the test.
        method (str): Nyström sampling method ('uniform' or other).
        k (int): Number of Nyström features.
        B (int): Number of permutations.

    Returns:
        output (int): 1 if null hypothesis is rejected, 0 otherwise.
        dt (float): Computation time in seconds.
        k (int): Number of Nyström features used.
    """
    start = time.time()  # Start computation timer
    rng = np.random.default_rng(seed)  # Random number generator

    ntot, d = Z.shape  # Total samples and feature dimension
    n = m = int(ntot / 2)  # Sample size per group

    # Compute Nyström feature mapping
    inds, _ = nys_inds(Z, k, method, bandwidth, seed) #(X, k, method, sigma, seed
    psi_Z = nystrom_features(Z, inds, bandwidth)

    # Perform permutation test
    I_1 = np.concatenate((np.ones(m), np.zeros(n)))
    I = np.tile(I_1, (B + 1, 1))
    I_X = independent_permutation(I, rng, axis=1)
    I_X[0] = I_1  # Keep original order for the first case
    I_Y = 1 - I_X

    # Compute test statistics
    bar_Z_B_piX = (1 / m) * I_X @ psi_Z
    bar_Z_B_piY = (1 / n) * I_Y @ psi_Z
    T = bar_Z_B_piX - bar_Z_B_piY
    V = np.sum(T ** 2, axis=1)

    rMMD2 = V[0]
    output, _ = decide(V, rMMD2, B, alpha)

    dt = time.time() - start  # Compute execution time
    return output, dt, k

# def NysMMDtest_split(X_train, X_test, seed=None, bandwidth=1, alpha=0.05, method='uniform', k=20, B=199, sample_at_once=False):
#     """
#     Performs the Nyström approximation MMD test.
    
#     Parameters:
#         X (array): Feature matrix of shape (n, d).
#         Y (array): Feature matrix of shape (n, d).
#         split_ratio (float in 0-1/2): ratio of training (to selecte centres) and test data.
#         seed (int, optional): Random seed for reproducibility.
#         bandwidth (float): Kernel bandwidth parameter.
#         alpha (float): Significance level for the test.
#         method (str): Nyström sampling method ('uniform' or other).
#         k (int): Number of Nyström features.
#         B (int): Number of permutations.

#     Returns:
#         output (int): 1 if null hypothesis is rejected, 0 otherwise.
#         dt (float): Computation time in seconds.
#         k (int): Number of Nyström features used.
#     """
#     start = time.time()  # Start computation timer
#     rng = np.random.default_rng(seed)  # Random number generator

#     # n, d = Z.shape

#     # n_x = int(n/2)
#     n_x_test = X_test.shape[0]//2

#     # X = Z[:n,:]
#     # Y = Z[n:,:]

#     # X_train_ids = rng.choice(n_x, n_train, replace=False)
#     # Y_train_ids = rng.choice(n_x, n_train, replace=False)

#     # X_train = X[X_train_ids,:]
#     # Y_train = Y[Y_train_ids,:]

#     # n_train, d = X_train.shape  # num of samples and feature dimension
#     # n_test, _ = X_test.shape  # num of samples and feature dimension

#     # n = m = int(X_test/2)

#     nystrom = NystromApproximator(X_train, k, bandwidth, seed, method=method, sample_at_once=sample_at_once)

#     psi_X = nystrom.fit_transform(X_test)

#     # Perform permutation test
#     I_1 = np.concatenate((np.ones(n_x_test), np.zeros(n_x_test)))
#     I = np.tile(I_1, (B + 1, 1))
#     I_X = independent_permutation(I, rng, axis=1)
#     I_X[0] = I_1  # Keep original order for the first case
#     I_Y = 1 - I_X

#     # Compute test statistics
#     bar_Z_B_piX = (1 / n_x_test) * I_X @ psi_X
#     bar_Z_B_piY = (1 / n_x_test) * I_Y @ psi_X
#     T = bar_Z_B_piX - bar_Z_B_piY
#     V = np.sum(T ** 2, axis=1)

#     MMD2 = V[0]
#     output, _ = decide(V, MMD2, B, alpha)

#     dt = time.time() - start  # Compute execution time
#     return output, dt, k
