from glob import glob
import numpy as np
from utils import extract_var_fromstring



# Similar function to load results, but specific for 'CG' data
def load_results(folder, methods=['uniform','rff','rlss', 'ctt']):

    results_dict = {}

    for method in methods:

        if method == 'fullrank':
            print(f"loading {method}")
            files = glob(folder+"/*/fullrank/results.npy", recursive=True)
            print(files)
            files = sorted(files, key=extract_var_fromstring)  # Sort based on var parameter in filename
            results = [np.load(el) for el in files]

            # Calculate average time, power, and number of features for each result
            time_pow_nfeat = np.asarray([(el[:,1].mean(axis=0), el[:,0].mean(axis=0), el[:,2].mean(axis=0)) for el in results])

            results_dict[method] = time_pow_nfeat

            if method == 'ctt':
                print(f"loading {method}")
                files = glob(folder+"/*/fullrank/results.npy", recursive=True)
                print(files)
                files = sorted(files, key=extract_var_fromstring)  # Sort based on var parameter in filename
                results = [np.load(el) for el in files]

                # Calculate average time, power, and number of features for each result
                time_pow_nfeat = np.asarray([(el[:,1].mean(axis=0), el[:,0].mean(axis=0), el[:,2].mean(axis=0)) for el in results])

                results_dict[method] = time_pow_nfeat

        else:  # Handle other methods
            print(f"loading {method}")
            files = glob(folder+"/*/"+method+"/results.npy", recursive=True)
            print(files)
            files = sorted(files, key=extract_var_fromstring)  # Sort based on var parameter in filename
            results = [np.load(el) for el in files]

            # Calculate average time, power, and number of features for each result
            time_pow_nfeat = np.asarray([(el[:,:,1].mean(axis=0), el[:,:,0].mean(axis=0), el[:,:,2].mean(axis=0)) for el in results])

            results_dict[method] = time_pow_nfeat

            

        # file_0_path = Path(files[0]) # read config from first file (they are all the same)
        # config = read_config_if_exists(file_0_path.parts[0]+'/'+file_0_path.parts[1]+'/'+'arguments.txt')

    return results_dict