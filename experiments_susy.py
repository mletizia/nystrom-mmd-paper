import numpy as np
import os
import argparse
from datetime import datetime

# Import custom utilities for Nyström permutation test, kernel parameter estimation, and dataset sampling
from tests import rMMDtest, NysMMDtest, MMDbtest
from samplers import sample_higgs_susy_dataset, read_data_susy
from utils import list_num_features, check_if_seeds_exist, median_pairwise

# Define constant for scaling
SQRT_2 = np.sqrt(2)

def main():

    parser = argparse.ArgumentParser()

    # Required named arguments
    parser.add_argument('--output_folder', type=str, default="./results", help='Folder where to store results. Default "./results".')   
    parser.add_argument('--tests', nargs='+', default=["uniform", "rlss", "rff"] , type=str, help='Input tests as a list, e.g.: fullrank uniform rlss rff')
    parser.add_argument('--alpha', default=0.05 , type=float, help='Level of the test')
    parser.add_argument('--B', default=199 , type=int, help='Number of permutations')
    parser.add_argument('--N', default=400 , type=int, help='Number of repetitions')
    parser.add_argument('--n', nargs='+', default=[1000, 2000, 4000, 8000, 12000, 16000, 20000, 40000] , type=int, help='List of sample sizes')
    parser.add_argument('--mix', default=0.05 , type=float, help='Proportion of class 1 data in the mixture')


    args = parser.parse_args()

    # Output folder
    of = args.output_folder

    # Specify which tests to perform
    which_tests = args.tests  # Test types to run

    # Parameters for statistical testing
    alpha = args.alpha  # Significance level of the test
    B = args.B  # Number of permutations in the permutation test
    n_tests = args.N  # Number of tests to perform on different subsamples

    # Parameters for dataset sampling
    sample_sizes = args.n  # Sample sizes
    lambda_mix = args.mix  # Proportion of class 1 in the mixture

    print("Susy experiments")  # Log the start of experiments

    # Print all arguments
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    # Load the full Higgs dataset
    X_all, Y_all = read_data_susy("/data/DATASETS/SUSY/Susy.mat")

    # Estimate the median pairwise distance for the RBF kernel parameter
    X_tune = sample_higgs_susy_dataset(X_all, Y_all, 1000, alpha_mix=lambda_mix, seed=None)
    sigmahat = median_pairwise(X_tune)  # Median pairwise distance as kernel bandwidth

    # Iterate over different sample sizes
    for n in sample_sizes:
        ntot = 2 * n
        # K = list_num_features(ntot)
        # print(f"Num. of features {K}")
        sqrt_n = int(np.sqrt(ntot))
        K = [10*sqrt_n]
        print(f"Num. of features {K}")

        # Define output folder for storing results
        output_folder = of+"/"+str(datetime.now().date())+f'/susy_B{B+1}_niter{n_tests}_mix{lambda_mix}/var{ntot}'
        os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it does not exist

        # # Save all arguments to a file
        # with open(output_folder + '/arguments.txt', 'w') as file:
        #     for arg, value in vars(args).items():
        #         file.write(f"{arg}: {value}\n")

        # Initialize arrays to store test results for each method
        if "fullrank" in which_tests:
            os.makedirs(output_folder + '/fullrank/')
            output_full = np.zeros(shape=(n_tests, 3))  # For full-rank tests

        if "uniform" in which_tests:
            os.makedirs(output_folder + '/uniform/')
            output_uni = np.zeros(shape=(n_tests, len(K), 3))  # For uniform sampling Nyström tests

        if "rlss" in which_tests:
            os.makedirs(output_folder + '/rlss/')
            output_rlss = np.zeros(shape=(n_tests, len(K), 3))  # For RLSS Nyström tests

        if "rff" in which_tests:
            os.makedirs(output_folder + '/rff/')
            output_rff = np.zeros(shape=(n_tests, len(K), 3))  # For random Fourier feature tests

        # Call function to check or generate seeds
        seeds = check_if_seeds_exist(output_folder, n_tests)

        # Perform tests over multiple iterations
        for test in range(n_tests):
            print(f"Test: {test + 1}/{n_tests} - ntot = {ntot}")  # Log the progress of tests

            # Assign the test seed
            test_seed = seeds[test]

            # Sample data subsets from the Higgs dataset for this iteration
            X = sample_higgs_susy_dataset(X_all, Y_all, n, alpha_mix=lambda_mix, seed=test_seed)

            # Perform full-rank permutation test if specified
            if "fullrank" in which_tests:
                print("Fullrank test")
                output_full[test, :] = MMDbtest(X, bw=sigmahat, seed=test_seed, alpha=0.05, B=199, plot=False)

            # Perform uniform Nyström-based permutation test if specified
            if "uniform" in which_tests:
                print("Uniform test")
                for i, k in enumerate(K):
                    output_uni[test, i, :] = NysMMDtest(X, seed=test_seed, bandwidth=sigmahat, alpha=0.05, method='uniform', k=k, B=B)

            # Perform recursive LSS Nyström-based permutation test if specified
            if "rlss" in which_tests:
                print("RLSS test")
                for i, k in enumerate(K):
                    output_rlss[test, i, :] = NysMMDtest(X, seed=test_seed, bandwidth=sigmahat, alpha=0.05, method='rlss', k=k, B=B)

            # Perform random Fourier feature-based permutation test if specified
            if "rff" in which_tests:
                print("RFF test")
                for i, k in enumerate(K):
                    output_rff[test, i, :] = rMMDtest(X, seed=test_seed, bandwidth=sigmahat*SQRT_2, alpha=alpha, R=k, B=B)

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