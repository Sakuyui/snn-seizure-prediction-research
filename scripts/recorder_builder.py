import numpy as np
from .feature_extractor import RecordConfiguration
from .features_func import *

def build_recorder_config(n_channels):
    config = RecordConfiguration()
    config.add_record_object("electrode_sd", lambda win: np.std(win, axis = 0))
    config.add_record_object('electrode_corr', lambda win: electrode_correlation(win, n_channels))
    config.add_record_object("autocorrelate", lambda win: calculate_auto_corr(win, 50))
    config.add_record_object("mutual_information", lambda win: mutual_information(win, time_step_lag=15))
    config.add_record_object("distribution_entropy", lambda win: distribution_entropy(win))
    config.add_record_object("network_entropy", lambda win: network_entropy(win, time_step_lag=15))
    return config
