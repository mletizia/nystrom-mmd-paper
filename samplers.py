import numpy as np
import h5py

def generate_correlated_gaussians(n, d, rho1, rho2, seed=None):
    """
    Generate two samples from correlated Gaussian distributions.
    
    Parameters
    ----------
    n : int
        Number of samples in each distribution.
    d : int
        Dimensionality of the data.
    rho1 : float
        Correlation coefficient between dimensions for the first Gaussian.
    rho2 : float
        Correlation coefficient between dimensions for the second Gaussian.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    
    Returns
    -------
    X : array_like, shape (2*n, d)
        Samples from the two correlated Gaussian distributions. First n samples from rho1, next n from rho2.
    """
    # Initialize random number generator with optional seed for reproducibility
    rng = np.random.default_rng(seed)
    
    # Define covariance matrices for the two distributions based on correlation coefficients
    cov1 = (1 - rho1) * np.eye(d) + rho1 * np.ones((d, d))  # Covariance matrix for first distribution
    cov2 = (1 - rho2) * np.eye(d) + rho2 * np.ones((d, d))  # Covariance matrix for second distribution

    # Means of the two distributions (zero mean)
    mean = np.zeros(d)
    
    # Create an array to hold both samples
    X = np.zeros((2 * n, d))  # First n rows for rho1, second n rows for rho2
    
    # Generate n samples from each Gaussian distribution using multivariate normal distribution
    X[:n] = rng.multivariate_normal(mean, cov1, size=n)  # First half (rho1)
    X[n:] = rng.multivariate_normal(mean, cov2, size=n)  # Second half (rho2)
    
    return X  # Return the generated samples


def read_data_higgs(file_name, reduced=0):
    """
    Loads the Higgs dataset and optionally reduces the feature space.
    
    Parameters
    ----------
    file_name : str
        Path to the data file.
    reduced : int, optional
        Option to reduce the feature space:
        - 0: No reduction (default).
        - 1: Reduce to a subset of features (jet phis).
        - 2: Further reduce to a smaller subset (first two jet phis).
    
    Returns
    -------
    X : np.ndarray
        Array of feature values.
    Y : np.ndarray
        Array of labels.
    """
    print(f"Loading Higgs data - reduced: {reduced}")
    # Open the HDF5 file and read the data
    with h5py.File(file_name, "r") as h5py_file:
        arr = np.array(h5py_file["X"], dtype=np.float64).T  # Transpose for correct orientation
    # Select which features to use based on reduction flag
    if reduced == 1:
        X = arr[:, [8,12,16,20]]  # Jet phis
    elif reduced == 2:
        X = arr[:, [8,12]]  # First two jet phis
    else:
        X = arr[:, 1:15]  # Default: low-level features
    Y = arr[:, 0]  # Labels are stored in the first column
    print("Done")
    return X, Y  # Return features and labels


def read_data_susy(file_name):
    """
    Loads the SUSY dataset.
    
    Parameters
    ----------
    file_name : str
        Path to the data file.
    
    Returns
    -------
    X : np.ndarray
        Array of feature values.
    Y : np.ndarray
        Array of labels.
    """
    print("Loading Susy data")
    # Open the HDF5 file and read the data
    with h5py.File(file_name, "r") as h5py_file:
        arr = np.array(h5py_file["X"], dtype=np.float64).T  # Transpose for correct orientation
    X = arr[:, 1:9]  # Low-level features
    Y = arr[:, 0]  # Labels are stored in the first column
    print("Done")
    return X, Y  # Return features and labels


def sample_higgs_susy_dataset(Z, Y, n, alpha_mix=1, seed=None):
    """
    Samples n instances from class 0 and n instances from a mixture of class 0 and class 1.
    
    Parameters
    ----------
    Z : np.ndarray
        Array of features.
    Y : np.ndarray
        Array of labels (0 for background, 1 for signal).
    n : int
        Number of instances to sample from each class.
    alpha_mix : float, optional (default=1)
        Proportion of class 1 in the mixture.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    sampled_Z : np.ndarray
        Sampled feature array.
    sampled_Y : np.ndarray
        Sampled label array with overridden labels (first n are 0, second n are 1).
    """
    if not (0 <= alpha_mix <= 1):
        raise ValueError("alpha_mix must be between 0 and 1.")  # Ensure alpha_mix is within valid range

    rng = np.random.default_rng(seed)  # Initialize RNG for reproducibility

    # Find indices of each class in the labels
    class_0_indices = np.where(Y == 0)[0]
    class_1_indices = np.where(Y == 1)[0]

    # Sample n instances from class 0
    if len(class_0_indices) < n:
        raise ValueError(f"Not enough instances in class 0 to sample {n}. Available: {len(class_0_indices)}")
    
    sampled_class_0_indices = rng.choice(class_0_indices, n, replace=False)  # Without replacement

    # Sample n instances for the mixed set with a specified proportion of class 0 and class 1
    num_class_1_in_mix = int(n * alpha_mix)
    num_class_0_in_mix = n - num_class_1_in_mix

    # Remove previously sampled class 0 indices from available pool
    reduced_class_0_indices = np.setdiff1d(class_0_indices, sampled_class_0_indices, assume_unique=True)

    # Ensure there are enough samples left to form the mixture
    if len(reduced_class_0_indices) < num_class_0_in_mix:
        raise ValueError(f"Not enough instances in class 0 for the mixture. Available: {len(reduced_class_0_indices)}")
    if len(class_1_indices) < num_class_1_in_mix:
        raise ValueError(f"Not enough instances in class 1 for the mixture. Available: {len(class_1_indices)}")

    # Sample class 0 and class 1 indices for the mixture
    mixed_class_0_indices = rng.choice(reduced_class_0_indices, num_class_0_in_mix, replace=False)
    mixed_class_1_indices = rng.choice(class_1_indices, num_class_1_in_mix, replace=False)
    
    # Combine the indices for the mixed set
    mixed_indices = np.concatenate([mixed_class_0_indices, mixed_class_1_indices])
    
    # shuffle background and signal indices
    rng.shuffle(mixed_indices)

    # Prepare output array for the sampled data
    sampled_Z = np.empty((2 * n, Z.shape[1]), dtype=Z.dtype)

    # Fill sampled feature array
    sampled_Z[:n] = Z[sampled_class_0_indices]  # First n are from class 0
    sampled_Z[n:] = Z[mixed_indices]  # Second n are from the mixture
    
    return sampled_Z  # Return the sampled feature set
