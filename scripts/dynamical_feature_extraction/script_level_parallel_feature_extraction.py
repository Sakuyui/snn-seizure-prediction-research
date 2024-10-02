import numpy as np
import os
import sys
import subprocess
import argparse
from data.dataset.dataset import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nf", "--n_fragment", type=int, default=30)
    args = parser.parse_args()
    
    n_fragment = args.n_fragment
    dataset_facade: EEGDatasetFacade = EEGDatasetFacade(dataset_base_path="./data")
    dataset: BaseEEGDataset = dataset_facade("ethz-ieeg")
    data = dataset.get_mne_data(['long', 1, 288]).resample(sfreq=300)
    
    freq = data.info['sfreq']
    numpy_data = data.get_data().T
    total_time_points = numpy_data.shape[0]

    avg_length = int(np.ceil(total_time_points / n_fragment))
    fragment_from = [i * avg_length for i in range(n_fragment)]
    
    lengths = [min(avg_length, total_time_points - avg_length * i) for i in range(n_fragment)]
    assert sum(lengths) == total_time_points

    processes = []
    for i, (fragment_start, length) in enumerate(zip(fragment_from, lengths)):
        print(f"Starting process for fragment {i}: from {fragment_start} with length {length}")
        
        # Create a subprocess for each fragment
        process = subprocess.Popen(['taskset', '-c', '%d' % i, 
                                     'python', '-m', 
                                     'scripts.dynamical_feature_extraction.feature_extraction_experiment', 
                                     '-pf', "%d" % fragment_start, 
                                     '-l', "%d" % length, 
                                     '-fi', "%d" % i], 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        processes.append(process)

    # Wait for all subprocesses to complete
    for process in processes:
        stdout, stderr = process.communicate()  # Capture output
        print(stdout.decode(), stderr.decode())  # Print the output for each process

    print("All processes have finished running.")
