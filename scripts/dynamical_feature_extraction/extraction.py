
from scripts.dynamical_feature_extraction import FeatureExtractor, RecordConfiguration, FeatureRecorder
from scripts.dynamical_feature_extraction import DynamicalFeatureRecorderBuilder, BaseRecorderBuilder, RecorderBuilder
from data.dataset.dataset import *


import pickle
import subprocess

# python -m scripts.dynamical_feature_extraction.feature_extraction_experiment
import cProfile
import sys, os
if __name__ == "__main__":
    dataset_facade : EEGDatasetFacade= EEGDatasetFacade(dataset_base_path="./data")
    dataset : BaseEEGDataset = dataset_facade("ethz-ieeg")
    index = int(sys.argv[1])
    print(f'index = {index}')

    with open(f'temp_{index}.pkl', 'rb') as f:
        recorder = pickle.load(f)
        print(recorder)
        recorder.prepared = True
        assert recorder is not None
        
    data = dataset.get_mne_data(['long', 1, 288])
    # data.resample(100, npad="auto")
    freq = data.info['sfreq']
    numpy_data = data.get_data().T
    del data
    
    feature_extractor = FeatureExtractor(freq, int(100))
    feature_extractor.profile_performance(False)
    
    
    print(recorder.record_path)
    recorder, performance = feature_extractor.do_extraction(numpy_data, recorder, time_limit=20, parallel=False)

    recorder.save_all("./data/features/", format = "record_%d.npy" % index)