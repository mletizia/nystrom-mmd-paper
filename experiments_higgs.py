import numpy as np
import os
import argparse
from datetime import datetime

# Import custom utilities for Nyström permutation test, kernel parameter estimation, and dataset sampling
from tests import rMMDtest, NysMMDtest, MMDbtest
from samplers import sample_higgs_susy_dataset, read_data_higgs
from utils import list_num_features, check_if_seeds_exist, median_pairwise

# Define constant for scaling
SQRT_2 = np.sqrt(2)

def main():


    parser = argparse.ArgumentParser()

    # Required named arguments
    parser.add_argument('--output_folder', type=str, default="./results", help='Folder where to store results. Default "./results".')   
    parser.add_argument('--tests', nargs='+', type=str, default=["uniform", "rlss", "rff"], help='Input tests as a list, e.g.: fullrank uniform rlss rff')    
    parser.add_argument('--alpha', default=0.05 , type=float, help='Level of the test')
    parser.add_argument('--B', default=199 , type=int, help='Number of permutations')
    parser.add_argument('--N', default=400 , type=int, help='Number of repetitions')
    parser.add_argument('--n', nargs='+', default=[500, 2500, 5000, 10000, 20000, 30000, 40000, 50000] , type=int, help='List of sample sizes')
    parser.add_argument('--mix', default=0.2 , type=float, help='Proportion of class 1 data in the mixture')
    parser.add_argument('--reduced', default=0 , type=int, help='Reduced: 0 (low-evel), 1 (4d), 2 (2d)')
    parser.add_argument('--K', nargs='+', type=int, help='List of features. E.g.: 5 30 100')


    args = parser.parse_args()

    # Output folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    of = args.output_folder+"/"+timestamp

    # Specify which tests to perform
    which_tests = args.tests  # Test types to run

    # Parameters for statistical testing
    alpha = args.alpha  # Significance level of the test
    B = args.B  # Number of permutations in the permutation test
    n_tests = args.N  # Number of tests to perform on different subsamples
    K_input = args.K

    # Parameters for dataset sampling
    sample_sizes = args.n  # Sample sizes
    lambda_mix = args.mix  # Proportion of class 1 in the mixture
    reduced = args.reduced  # Reduction mode for dataset

    print("Higgs experiments")  # Log the start of experiments

    # Print all arguments
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    # Load dataset
    X_all, Y_all = read_data_higgs("/data/DATASETS/HIGGS_UCI/Higgs.mat", reduced=reduced)

    # Estimate kernel bandwidth
    X_tune = sample_higgs_susy_dataset(X_all, Y_all, 1000, alpha_mix=lambda_mix, seed=None)
    sigmahat = median_pairwise(X_tune)

    # Iterate over different sample sizes
    for n in sample_sizes:
        ntot = 2 * n
        if K_input==None: K = list_num_features(ntot)
        else: K = K_input
        print(f"Num. of features {K}")

        # Define output folder
        output_folder = of+f'/higgs_B{B+1}_niter{n_tests}_mix{lambda_mix}_reduced{reduced}/var{ntot}'
        os.makedirs(output_folder, exist_ok=True)

        # Save all arguments to a file
        with open(output_folder + '/arguments.txt', 'w') as file:
            for arg, value in vars(args).items():
                file.write(f"{arg}: {value}\n")

        # Initialize arrays for storing test results
        if "fullrank" in which_tests:
            os.makedirs(output_folder + '/fullrank/')
            output_full = np.zeros(shape=(n_tests, 3))

        if "uniform" in which_tests:
            os.makedirs(output_folder + '/uniform/')
            output_uni = np.zeros(shape=(n_tests, len(K), 3))

        if "rlss" in which_tests:
            os.makedirs(output_folder + '/rlss/')
            output_rlss = np.zeros(shape=(n_tests, len(K), 3))

        if "rff" in which_tests:
            os.makedirs(output_folder + '/rff/')
            output_rff = np.zeros(shape=(n_tests, len(K), 3))

        # Generate or retrieve seeds
        seeds = check_if_seeds_exist(output_folder, n_tests)

        # Run tests
        for test in range(n_tests):
            print(f"Test: {test + 1}/{n_tests} - ntot = {ntot}")
            test_seed = seeds[test]
            X = sample_higgs_susy_dataset(X_all, Y_all, n, alpha_mix=lambda_mix, seed=test_seed)

            if "fullrank" in which_tests:
                print("Fullrank test")
                output_full[test, :] = MMDbtest(X, n, n, bw=sigmahat, seed=test_seed, alpha=0.05, B=199, plot=False)

            if "uniform" in which_tests:
                print("Uniform test")
                for i, k in enumerate(K):
                    output_uni[test, i, :] = NysMMDtest(X, n, n, seed=test_seed, bandwidth=sigmahat, alpha=0.05, method='uniform', k=k, B=B)

            if "rlss" in which_tests:
                print("RLSS test")
                for i, k in enumerate(K):
                    output_rlss[test, i, :] = NysMMDtest(X, n, n, seed=test_seed, bandwidth=sigmahat, alpha=0.05, method='rlss', k=k, B=B)

            if "rff" in which_tests:
                print("RFF test")
                for i, k in enumerate(K):
                    output_rff[test, i, :] = rMMDtest(X, n, n, seed=test_seed, bandwidth=SQRT_2*sigmahat, alpha=alpha, R=k, B=B)

        # Save test results
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