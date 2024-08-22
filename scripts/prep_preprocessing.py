from pyprep.find_noisy_channels import NoisyChannels
import numpy as np

def noise_channel_detection(raw_eeg):
    noisy_detector = NoisyChannels(raw_eeg, random_state=None)
    noisy_detector.find_bad_by_nan_flat()
    noisy_detector.find_bad_by_correlation(correlation_secs=1.0, correlation_threshold=0.4, frac_bad=0.01)
    noisy_detector.find_bad_by_deviation(deviation_threshold=5.0)
    
    noisy_detector.find_bad_by_ransac(n_samples=50,
        sample_prop=0.25,
        corr_thresh=0.75,
        frac_bad=0.4,
        corr_window_secs=5.0,
        channel_wise=False,
        max_chunk_size=None,
    )
    
    return noisy_detector
    
def fit(self, raw_eeg, random_state):
    noisy_detector = NoisyChannels(raw_eeg, random_state=random_state)        
    _perform_reference(raw_eeg)
    return self

def _perform_reference(raw, prep_params, random_state, **ransac_settings):
    pass


