import numpy as np
import os, json
import argparse
from datetime import datetime

# Import custom utilities for Nyström permutation test, kernel parameter estimation, and dataset sampling
from tests import rMMDtest, NysMMDtest, MMDbtest
from samplers import generate_correlated_gaussians
from utils import check_if_seeds_exist, standardize_data, median_pairwise

# Define constant for scaling
SQRT_2 = np.sqrt(2)

def main():

    parser = argparse.ArgumentParser() 

    # Required named arguments
    parser.add_argument('--output_folder', type=str, default="./results", help='Folder where to store results. Default "./results".')   
    parser.add_argument('--tests', nargs='+', type=str, default=["uniform", "rlss", "rff"], help='Input tests as a list. For example: fullrank uniform rlss rff')    
    parser.add_argument('--alpha', default=0.05 , type=float, help='Level of the test')
    parser.add_argument('--B', default=199 , type=int, help='Number of permutations')
    parser.add_argument('--N', default=400 , type=int, help='Number of repetitions')
    parser.add_argument('--n', default=2500 , type=int, help='Sample size')
    parser.add_argument('--d', default=3 , type=int, help='Dimensions')
    parser.add_argument('--rho1', default=0.5 , type=float, help='Values of rho1')
    parser.add_argument('--rho2', nargs='+', type=float, default=[0.51, 0.54, 0.57, 0.60, 0.63, 0.66, 0.69], help='List of values of rho2.')
    parser.add_argument('--K', nargs='+', type=int, default=[14, 28, 42, 56, 70, 140, 350, 500], help='List of num. of features.')

    args = parser.parse_args()

    # Output folder
    of = args.output_folder
    os.makedirs(of, exist_ok=False)

    # Specify which tests to perform
    which_tests = args.tests  # Test types to run

    # Parameters for statistical testing
    alpha = args.alpha  # Significance level of the test
    B = args.B  # Number of permutations in the permutation test
    n_tests = args.N  # Number of tests to perform on different subsamples

    # Parameters for dataset sampling
    n = args.n  # Size of each sample. Total size = 2 * sample_size
    d = args.d  # Number of dimensions
    RHO2 = args.rho2  # Correlation values to test

    # Compute total sample size
    ntot = 2 * n

    # Define feature list
    K = args.K

    # save parameters
    params = vars(args)
    json_path = os.path.join(of, "params.json")
    with open(json_path, "w") as f:
        json.dump(params, f, indent=4)
 

    print("CG experiments")  # Log the start of experiments

    # Print all arguments
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print(f"Saved parameters to {json_path}")

    # Iterate over different correlation values
    for rho2 in RHO2:
        # Estimate the median pairwise distance for the RBF kernel parameter
        X_tune = generate_correlated_gaussians(500, d, rho1=args.rho1, rho2=rho2, seed=None)
        X_tune = standardize_data(X_tune)
        sigmahat = median_pairwise(X_tune)  # Compute kernel bandwidth

        # Define output folder for storing results
        output_folder = of+"/"+str(datetime.now().date())+f'/cg_ntot{ntot}_B{B+1}_niter{n_tests}/var{rho2}'
        os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it does not exist

        # Initialize arrays to store test results for each method
        if "fullrank" in which_tests:
            os.makedirs(output_folder + '/fullrank/')
            output_full = np.zeros(shape=(n_tests, 3))  # For full-rank tests

        if "uniform" in which_tests:
            os.makedirs(output_folder + '/uniform/')
            output_uni = np.zeros(shape=(n_tests, len(K), 3))  # For uniform sampling Nyström tests

        if "rlss" in which_tests:
            os.makedirs(output_folder + '/rlss/')
            output_rlss = np.zeros(shape=(n_tests, len(K), 3))  # For recursive Nyström tests

        if "rff" in which_tests:
            os.makedirs(output_folder + '/rff/')
            output_rff = np.zeros(shape=(n_tests, len(K), 3))  # For random Fourier features tests

        # Call function to check or generate seeds
        seeds = check_if_seeds_exist(output_folder, n_tests)

        # Perform tests over multiple iterations
        for test in range(n_tests):
            print(f"Test: {test + 1}/{n_tests} - ntot = {ntot}")  # Log test progress

            # Assign the test seed
            test_seed = seeds[test]

            # Generate correlated Gaussian samples and standardize them
            X = generate_correlated_gaussians(n, d, rho1=0.5, rho2=rho2, seed=test_seed)
            X = standardize_data(X)

            # Perform full-rank permutation test if specified
            if "fullrank" in which_tests:
                print("Fullrank test")
                output_full[test, :] = MMDbtest(X, n, n, bw=sigmahat, seed=test_seed, B=B, plot=False)

            # Perform uniform Nyström-based permutation test if specified
            if "uniform" in which_tests:
                print("Uniform test")
                for i, k in enumerate(K):
                    output_uni[test, i, :] = NysMMDtest(X, n, n, seed=test_seed, bandwidth=sigmahat, alpha=0.05, method='uniform', k=k, B=B)

            # Perform recursive LSS Nyström-based permutation test if specified
            if "rlss" in which_tests:
                print("RLSS test")
                for i, k in enumerate(K):
                    output_rlss[test, i, :] = NysMMDtest(X, n, n, seed=test_seed, bandwidth=sigmahat, alpha=0.05, method='rlss', k=k, B=B)

            # Perform Random Fourier Features-based test if specified
            if "rff" in which_tests:
                print("RFF test")
                for i, k in enumerate(K):
                    output_rff[test, i, :] = rMMDtest(X, n, n, seed=test_seed, bandwidth=SQRT_2 * sigmahat, alpha=alpha, R=k, B=B)

        # Save results for each test type to the corresponding subdirectory
        if "fullrank" in which_tests:
            np.save(output_folder + '/fullrank/results.npy', output_full)
        if "uniform" in which_tests:
            np.save(output_folder + '/uniform/results.npy', output_uni)
        if "rlss" in which_tests:
            np.save(output_folder + '/rlss/results.npy', output_rlss)
        if "rff" in which_tests:
            np.save(output_folder + '/rff/results.npy', output_rff)


# Main execution block
if __name__ == "__main__":

    main()
