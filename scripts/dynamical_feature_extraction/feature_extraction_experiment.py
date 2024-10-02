import numpy as np
import os, sys
from scripts.dynamical_feature_extraction import FeatureExtractor, RecordConfiguration, FeatureRecorder
from scripts.dynamical_feature_extraction import DynamicalFeatureRecorderBuilder, BaseRecorderBuilder
from scripts.dynamical_feature_extraction.recorder_builder import RecorderBuilder
from data.dataset.dataset import *

import argparse

# python -m scripts.dynamical_feature_extraction.feature_extraction_experiment
parser = argparse.ArgumentParser()
parser.add_argument("-pf", "--process_from", type=int, default=0)
parser.add_argument("-l", "--length", type=int, default=-1)
parser.add_argument("-fi", "--fragment_index", type=int, default=0)

win_size = 200

if __name__ == "__main__":
    args = parser.parse_args()

    dataset_facade : EEGDatasetFacade= EEGDatasetFacade(dataset_base_path="./data")
    dataset : BaseEEGDataset = dataset_facade("ethz-ieeg")
    
    data = dataset.get_mne_data(['long', 1, 288]).resample(sfreq=300)
    
    freq = data.info['sfreq']
    numpy_data = data.get_data().T
    del data
    process_from = args.process_from
    length = numpy_data.shape[0] if args.length == -1 else args.length
    numpy_data = numpy_data[:length + win_size] if process_from == 0 else numpy_data[process_from - win_size: process_from - win_size + length]
    print(numpy_data.shape)
    config = RecordConfiguration()
    n_channels = numpy_data.shape[1]

    print(" Build Recorder Configuration...")
    recorder_configuration_builder = DynamicalFeatureRecorderBuilder()
    recorder_configuration = recorder_configuration_builder.build_recorder_config(n_channels=n_channels)
    
    recorder_builder = RecorderBuilder()
    recorder: FeatureRecorder = recorder_builder.build_record(recorder_configuration, enable_pca=4, parallelism_detection=False, separate_parallelizable_records=False)
    
    print(f"Feature Record Initialized, record path = {recorder.record_path}")

    feature_extractor = FeatureExtractor(freq, int(win_size))
    feature_extractor.profile_performance(True)

    recorder, performance = feature_extractor.do_extraction(numpy_data, recorder, time_limit=-1, parallel=False)
    
    print(performance)
    
    recorder.save_all("./data/features/", format = ("record_%d_" % args.fragment_index) + "%s.npy")