from rp_segmentation_module import InfinitePhaseSpaceReonstructionBasedSegmentGenerator, FiniteTimeDelaySegmentGenerator, FiniteTimeDelayEEGSegmentGenerator

import numpy as np
import re
import os, sys
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--use_microstate", type=bool, default=True)
parser.add_argument("-cf", "--configuration-file", type=str, default="./epilepsy_dataset_phase_space_reconstruction.json")
parser.add_argument("-i", "--index-only", type=bool, default=False)
parser.add_argument("-s", "--split_normal_seizure", type=bool, default=False)
args = parser.parse_args()

with open(args.configuration_file) as f:
    configuration_content = f.read()
    dict_args = json.loads(configuration_content)
    f.close()

sys.path.append("../../data/dataset")
sys.path.append("../../microstate_lib/code")
from dataset import *

dataset_base_path = dict_args.get('dataset_base_path', '../data')
dataset_facade = EEGDatasetFacade(dataset_base_path=dataset_base_path)
dataset = dataset_facade(dict_args.get("dataset_name", ""))

def split_data_into_normal_and_seizures(data, record_ids):
    seizures = dataset.get_seizure_ranges_in_time_offsets(record_ids)
    sorted(seizures, key=lambda x: x[0])
    splitted_data = {
        'seizures': [],
        'normal': []
    }
    for seizure_annotation in seizures:
        splitted_data['seizures'].append(data[seizure_annotation[0], seizure_annotation[1] + 1])
    previous_position = 0
    for seizure_annotation in seizures:
        splitted_data['normal'].append(data[previous_position, seizure_annotation[0]])
        previous_position = seizure_annotation[1] + 1
    return splitted_data



corpus_storage_base_path = dict_args['corpus_storage_base_path']
microstate_storage_base_path = dict_args['microstate_storage_base_path']
if not os.path.exists(corpus_storage_base_path):
        os.makedirs(corpus_storage_base_path, exist_ok=True)
        
if args.use_microstate:
    sids = dict_args['sids']     
    
    for index, sid in enumerate(sids):
        data = dataset.get_eeg_microstate_sequence(sid, dict_args['microstate_filename_form'], reduce_to_segments = True)
        data = split_data_into_normal_and_seizures(data, args['merged_record_ids'][index])

        delay = dict_args['delay']
        n_states = dict_args['n_states']
        if not args.split_normal_seizure:
            segment_generator = FiniteTimeDelaySegmentGenerator(data=data, time_delay=delay, n_states=n_states, cut=dict_args['cut'])
            if args.index_only:
                segments = segment_generator.calculate_recurrent_plot_points()
            else:
                segments = segment_generator.calculate_recurrent_segments()
            np.save(os.path.join(corpus_storage_base_path, f'{sid}_d{delay}_s{n_states}.npy'), np.array(segments, dtype='object'), allow_pickle=True)
        else:
            for seizure_data_index, seizure_data in enumerate(data['seizures']):
                segment_generator = FiniteTimeDelaySegmentGenerator(data=seizure_data, time_delay=delay, n_states=n_states, cut=dict_args['cut'])
                if args.index_only:
                    segments = segment_generator.calculate_recurrent_plot_points()
                else:
                    segments = segment_generator.calculate_recurrent_segments()
                np.save(os.path.join(corpus_storage_base_path, f'seizure_{seizure_data_index}_{sid}_d{delay}_s{n_states}.npy'), np.array(segments, dtype='object'), allow_pickle=True)
            for normal_data_index, seizure_data in enumerate(data['normal']):
                segment_generator = FiniteTimeDelaySegmentGenerator(data=seizure_data, time_delay=delay, n_states=n_states, cut=dict_args['cut'])
                if args.index_only:
                    segments = segment_generator.calculate_recurrent_plot_points()
                else:
                    segments = segment_generator.calculate_recurrent_segments()
                np.save(os.path.join(corpus_storage_base_path, f'normal_{seizure_data_index}_{sid}_d{delay}_s{n_states}.npy'), np.array(segments, dtype='object'), allow_pickle=True)
else:
    raise NotImplementedError
