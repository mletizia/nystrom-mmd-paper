import re, os
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import pdist

import re, ast
from pathlib import Path

# Function to estimate the width of the Gaussian kernel using pairwise distances
def median_pairwise(data):
    # this function estimates the width of the gaussian kernel.
    # use on a small sample (standardize first if necessary)
    pairw = pdist(data)  # Pairwise distance calculation (Euclidean by default)
    return np.median(pairw)  # Return the median of pairwise distances

# Function to check if a 'seeds.npy' file exists and load or generate seeds
def check_if_seeds_exist(output_folder, n_tests):
    """
    Checks if a seeds.npy file exists in the given output folder.
    - If it exists, loads and returns the seeds.
    - If it doesn't exist, generates new seeds, saves them, and returns them.

    Args:
        output_folder (str): Path to the folder where seeds.npy should be stored.
        n_tests (int): Number of test iterations.

    Returns:
        np.ndarray: Array of seeds.
    """
    seed_file = os.path.join(output_folder, "seeds.npy")
    
    # If seed file exists, load it
    if os.path.exists(seed_file):
        print(f"Loading existing seeds from {seed_file}")
        seeds = np.load(seed_file)
    else:
        # If seed file doesn't exist, generate new seeds and save them
        print(f"Generating new seeds for {n_tests} test iterations")
        seeds = generate_seeds(n_tests)  
        np.save(seed_file, seeds)  # Save for reproducibility
        print(f"Saved new seeds to {seed_file}")

    return seeds

# Function to generate 'n' random seeds using a given or default seed
def generate_seeds(n, seed=None):
    rng = np.random.default_rng(seed)  # Create a random number generator with the initial seed
    seeds = rng.integers(0, 2**32, size=n, dtype=np.uint32)  # Generate n random seeds
    return seeds

# Function to generate a list of feature counts based on the square root of n
def list_num_features(n):
    # Calculate sqrt(n)
    sqrt_n = int(np.sqrt(n))

    # Generate three integers between 0 and sqrt(n) (exclusive)
    before_sqrt = np.linspace(0, sqrt_n, num=5, endpoint=False, dtype=int)[1:]

    # Combine all numbers into a single sorted list
    result = np.concatenate((before_sqrt, [sqrt_n], [2*sqrt_n], [5*sqrt_n], [10*sqrt_n]))

    return result

# Function to generate a list of feature counts based on the square root of n
def list_num_features_fast(n):
    # Calculate sqrt(n)
    sqrt_n = int(np.sqrt(n))

    # Combine all numbers into a single sorted list
    result = np.concatenate(([sqrt_n//2], [sqrt_n], [2*sqrt_n], [5*sqrt_n]))

    return result

# Function to calculate the Wilson score interval for binomial proportions
def wilson_score_interval(p, n, confidence_level=0.95):
    """
    Calculate the Wilson score confidence interval for a binomial proportion.

    Parameters:
    p (float): Proportion of successes (rate of success).
    n (int): Number of trials.
    confidence_level (float): Confidence level for the interval (default is 0.95 for 95% confidence).

    Returns:
    tuple: Lower and upper bounds of the Wilson score interval.
    """
    if n == 0:
        raise ValueError("Number of trials (n) must be greater than 0.")

    if not (0 <= p <= 1):
        raise ValueError("Proportion (p) must be between 0 and 1.")

    # Z-value for the given confidence level
    alpha = 1 - confidence_level
    z = norm.ppf(1 - alpha / 2)

    # Wilson score calculation
    denominator = 1 + (z ** 2 / n)
    center = (p + (z ** 2 / (2 * n))) / denominator
    margin = (z * np.sqrt((p * (1 - p) / n) + (z ** 2 / (4 * n ** 2)))) / denominator

    lower_bound = max(0, center - margin)
    upper_bound = min(1, center + margin)

    return lower_bound, upper_bound

# Function to calculate the Wilson score interval with an additional power interval return
def power_interval(p, n, confidence_level=0.95):
    """
    Calculate the Wilson score confidence interval for a binomial proportion.

    Parameters:
    p (float): Proportion of successes (rate of success).
    n (int): Number of trials.
    confidence_level (float): Confidence level for the interval (default is 0.95 for 95% confidence).

    Returns:
    tuple: Proportion, lower and upper bounds of the Wilson score interval.
    """
    if n == 0:
        raise ValueError("Number of trials (n) must be greater than 0.")

    if not (0 <= p <= 1):
        raise ValueError("Proportion (p) must be between 0 and 1.")

    # Z-value for the given confidence level
    alpha = 1 - confidence_level
    z = norm.ppf(1 - alpha / 2)

    # Wilson score calculation
    denominator = 1 + (z ** 2 / n)
    center = (p + (z ** 2 / (2 * n))) / denominator
    margin = (z * np.sqrt((p * (1 - p) / n) + (z ** 2 / (4 * n ** 2)))) / denominator

    lower_bound = max(0, center - margin)
    upper_bound = min(1, center + margin)

    return p, lower_bound, upper_bound

# Function to decide whether to reject the null hypothesis based on a test statistic and null distribution
def decide(H0, t_obs, B, alpha):
    """
    Perform hypothesis testing decision.

    Args:
    H0 (array): Null distribution of test statistics.
    t_obs (float): Observed test statistic.
    B (int): Number of bootstrap samples.
    alpha (float): Significance level.

    Returns:
    output (int): 1 if null hypothesis is rejected, 0 otherwise.
    thr (float): Threshold value for decision.
    """
    # Sort null distribution
    H0_sorted = np.sort(H0)
    
    # Determine threshold based on significance level
    thr_ind = int(np.ceil((B + 1) * (1 - alpha))) - 1  # Index of alpha-level threshold
    thr = H0_sorted[thr_ind]

    # Handle cases where observed test statistic equals the threshold
    if t_obs == thr:
        greater = np.sum(H0_sorted > t_obs)  # Count values greater than t_obs
        equal = np.sum(H0_sorted == t_obs)  # Count values equal to t_obs
        a_prob = (alpha * (B + 1) - greater) / equal  # Adjusted probability

        # Randomly decide outcome based on adjusted probability
        output = np.random.default_rng(seed=None).choice([1, 0], p=[a_prob, 1 - a_prob])
    else:
        # Return 1 (reject null) if t_obs > thr, otherwise 0 (fail to reject null)
        output = int(t_obs > thr)

    return output, thr

# Function to perform independent permutations on a given array along a specified axis
def independent_permutation(I, rng, axis=1):
    """
    Perform independent permutations along a specified axis.
    Parameters:
        I: array_like
            Input array to permute.
        rng: np.random.Generator
            Random number generator.
        axis: int
            Axis along which to apply independent permutations (0 or 1).
    Returns:
        permuted: array_like
            The array with independent permutations applied along the specified axis.
    """
    permuted = np.empty_like(I)
    if axis == 1:  # Row-wise independent permutation
        for i in range(I.shape[0]):
            permuted[i] = rng.permutation(I[i])
    elif axis == 0:  # Column-wise independent permutation
        for i in range(I.shape[1]):
            permuted[:, i] = rng.permutation(I[:, i])
    else:
        raise ValueError("Axis must be 0 or 1.")
    return permuted


# Function to standardize data (zero mean and unit variance)
def standardize_data(data):
    """
    Standardize the given data (zero mean and unit variance).
    
    Parameters:
        data (numpy.ndarray): The data to be standardized, with samples as rows and features as columns.
    
    Returns:
        numpy.ndarray: Standardized data with the same shape as input.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data




def return_parameters(s: str) -> dict:
    # Extract the last part after the last "/"
    s = s.split("/")[-1]
    
    # Split the string by underscores to handle dataset-style strings
    parts = s.split("_")
    dataset_name = parts[0]  # The first part is the dataset name
    params = {}

    # Iterate through the remaining parts, ensuring each part is a key-value pair
    for part in parts[1:]:
        # Try to split into key and value
        match = re.match(r"([a-zA-Z]+)([0-9\.]+)", part)
        if match:
            key, value = match.groups()
            try:
                # Convert numerical values to int or float
                if '.' in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                params[key] = value  # Keep as a string if conversion fails
        else:
            # If no match, we assume the part is a malformed key-value pair and keep it as a string
            params[part] = part
    
    return {"dataset": dataset_name, **params}


# Helper function to extract 'ntot' value from filenames
def extract_ntot_fromstring(s):
    match = re.search(r'ntot(\d+)_', s)  # Find digits between 'n' and '_'
    return int(match.group(1)) if match else 0  # Convert to integer for proper sorting

# Helper function to extract 'rho' value from filenames
def extract_rho_fromstring(s):
    match = re.search(r'rho([\d\.]+)', s)  # Find digits between 'n' and '_'
    return float(match.group(1)) if match else 0.0  # Convert to float for proper sorting

def extract_var_fromstring(s):
    match = re.search(r'var([\d\.]+)', s)  # Find digits after 'var'
    var_value = float(match.group(1)) if match else 0.0  # Convert to float for proper sorting
    return var_value  # Return just the var value for sorting


def read_config_if_exists(file_path):
    config = {}
    file = Path(file_path)
    if file.is_file():
        with open(file_path, 'r') as file:
            for line in file:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        # Safely evaluate lists, numbers, etc.
                        config[key] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        # Fallback for plain strings
                        config[key] = value
    return config
