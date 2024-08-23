from rp_segmentation_module import InfinitePhaseSpaceReonstructionBasedSegmentGenerator, FiniteTimeDelaySegmentGenerator, FiniteTimeDelayEEGSegmentGenerator

import pyprep
import numpy as np
import mne
import os, sys
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--use_microstate", type=bool, default=True)
parser.add_argument("-d", "--delay", type=int, default=3)
parser.add_argument("-s", "--states", type=int, default=4)
parser.add_argument("-p", "--save-path", type=str, default="./sentences.npy")

args = parser.parse_args()

sys.path.append("../data/dataset")
sys.path.append("..//microstate_lib/code")

from dataset import *

dataset_base_path = "../data"
dataset_facade = EEGDatasetFacade(dataset_base_path=dataset_base_path)
dataset = dataset_facade("epileptic_eeg_dataset")

def to_segment_sequence(microstate_sequence):
    pre_state = -1
    segment_sequence = []
    for i in range(len(microstate_sequence)):
        state = microstate_sequence[i]
        if pre_state < 0:
            pre_state = state
        elif microstate_sequence[i] != pre_state:
            segment_sequence.append(pre_state)
            pre_state = state
    return np.array(segment_sequence)

if args.use_microstate:
    data = np.load("../data/sEEG/epileptic_eeg_dataset/[seg-[prep-asr]]person_10_states4_gev_0.8419650374810485.npy")
    data = to_segment_sequence(data)
else:
    data = dataset.get_merge_numpy_data([[10, 1], [10, 2]]).T


segment_generator = FiniteTimeDelaySegmentGenerator(data=data, time_delay=args.delay, n_states=args.states)
segments = segment_generator.calculate_recurrent_plot_points()
np.save(args.save_path + "", segments)
