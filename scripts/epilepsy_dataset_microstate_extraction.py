import numpy as np
import mne
import os, sys
from datetime import datetime
sys.path.append("../data/dataset")
sys.path.append("..//microstate_lib/code")
from dataset import *
import eeg_recording
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-pb", "--path-base", default="../data")
parser.add_argument("-dn", "--database-name", default="epileptic_eeg_dataset")
parser.add_argument("-dic", "--database-index-configuration", default=None)
parser.add_argument("-nml", "--number-microstate-least", default=4)
parser.add_argument("-nmm", "--number-microstate-most", default=16)
parser.add_argument("-ki", "--kmeans-iterations", default=100)
parser.add_argument("-sthr", "--stop-threshold", default=0.025)
parser.add_argument("-s4", "--store-microstates-n4", type=bool, default=True)
parser.add_argument("-sp", "--store-preprocessed", type=bool, default=True)
parser.add_argument("-ss", "--store-segmentation", type=bool, default=True)

args = parser.parse_args()


dataset_base_path = args.path_base
dataset_facade = EEGDatasetFacade(dataset_base_path=dataset_base_path)
dataset_name = args.database_name
dataset = dataset_facade(dataset_name)


# ["prep", {
#             "montage": "standard_1020",
#             "prep_params":{
#                 "ref_chs": "eeg",
#                 "reref_chs": "eeg",
#                 "line_freqs":[],
#             },
#             "reference_args": {
#                 "correlation_secs": 1.0, 
#                 "correlation_threshold": 0.4, 
#                 "frac_bad": 0.01
#             }
#             }]

if args.database_index_configuration == None:
    record_configuration = {
    "indexes":{
        # "10": [[10, 1], [10, 2]], 
        # "11": [[11, 1], [11, 2], [11, 3], [11, 4]], 
        "12": [[12, 1], [12, 2], [12, 3]],
        "13": [[13, 1], [13, 2], [13, 3], [13, 4]],
        "14": [[14, 1]],
        "15": [[15, 1], [15, 2], [15, 3], [15, 4]]
    },
    "preprocessings":{
        "pipeline":[["drop_channels", {'ch_names': ["ECG EKG", "Manual"], 'on_missing': 'warn'}], 
                    
["prep", {
            "montage": "standard_1020",
            "prep_params":{
                "ref_chs": "eeg",
                "reref_chs": "eeg",
                "line_freqs":[],
            },
            "reference_args": {
                "correlation_secs": 1.0, 
                "correlation_threshold": 0.4, 
                "frac_bad": 0.01
            }
            }],["asr", {"cutoff": 30}], ["average_reference", {}], ["min_max_nor", {}]],
        "post_merge_pipeline": [["average_reference", {}], ["min_max_nor", {}]]
    }
    }
else:
    with open(args.database_index_configuration) as f: 
        data = f.read() 
        record_configuration = json.loads(data)
        f.close()

record_indexes = record_configuration['indexes']
preprocessing_pipeline = record_configuration['preprocessings']['pipeline']
post_merge_pipeline =  record_configuration['preprocessings']['post_merge_pipeline']
microstate_search_range = (args.number_microstate_least, args.number_microstate_most)
n_iters = args.kmeans_iterations

stop_delta_threshold = args.stop_threshold
store_4_microstates = args.store_microstates_n4
save_preprocessed_data = args.store_preprocessed
save_segmentation = args.store_segmentation

store_base_path = dataset.base_path
global start_time
global end_time
start_time = -1
end_time = -1

def begin_timing():
    global start_time
    start_time = datetime.now()
    
def end_timing():
    global end_time
    end_time = datetime.now()
    
def report_execution_time(event = ""):
    end_timing()
    print('[%s] Time Consumption: {}'.format(event, end_time - start_time))

def store(maps, segmentation, gev, preprocessing_desc, person_id):
    n_states = maps.shape[0]
    save_map_file_name = f"[{preprocessing_desc}]person_{person_id}_states{n_states}_gev_{gev}.npy"

    np.save(os.path.join(store_base_path, save_map_file_name), maps)
    if save_segmentation:
        save_segmentation_file_name = f"[seg-{preprocessing_desc}]person_{person_id}_states{n_states}_gev_{gev}.npy"
        np.save(os.path.join(store_base_path, save_segmentation_file_name), segmentation)

load_preprocessing = set([])

# from pyprep.prep_pipeline import PrepPipeline

# prep_params = {
#     "ref_chs": "eeg",
#     "reref_chs": "eeg",
#     "line_freqs": [],
# }

montage_kind = "standard_1020"
montage = mne.channels.make_standard_montage(montage_kind)
sys.path.append("../")
from data.dataset.preprocessing import PreprocessingController

for person_index in record_indexes:
    print(f"Train microstates for person {person_index}")
    record_index_list = record_indexes[person_index]
    
    # preprocessings = [('prep', {'correlation_threshold': 0.4, 'frac_bad': 0.1}), ('asr')]

    if person_index not in load_preprocessing:
        data_count = len(record_index_list)
        results = []
        block_size = 1
        for slice_begin in range(0, data_count, block_size):
            data = dataset.get_merge_mne_data(record_index_list[slice_begin: slice_begin + block_size])
            
            data.rename_channels({ch_name: ch_name.replace("EEG ", "").replace("-Ref", "") for ch_name in data.ch_names})
            
            #! --- preprocessing ---
            for index, preprocessing_pipeline_item in enumerate(preprocessing_pipeline):
                print(f"[Preprocessing {index}: {slice_begin // block_size + 1}/{int(np.ceil(data_count / block_size))}]... name = {preprocessing_pipeline_item[0]}")
                preprocessing_name = preprocessing_pipeline_item[0]
                preprocessing_arguments = preprocessing_pipeline_item[1]
                PreprocessingController.preprocessing(data, preprocessing_name, preprocessing_arguments)
            
            results.append(data)
        data = mne.concatenate_raws(results)
        del results
        
        for index, preprocessing_pipeline_item in enumerate(post_merge_pipeline):
            print(f"[Post Merging Preprocessing {index}: {slice_begin // block_size + 1}/{int(np.ceil(data_count / block_size))}]... name = {preprocessing_pipeline_item[0]}")
            preprocessing_name = preprocessing_pipeline_item[0]
            preprocessing_arguments = preprocessing_pipeline_item[1]
            PreprocessingController.preprocessing(data, preprocessing_name, preprocessing_arguments)
        
        if save_preprocessed_data:
            mne.export.export_raw(os.path.join(store_base_path, f'[preprocessed_prep_asr]p{person_index}.edf'), data, overwrite=True)
    else:
        print(f"Load preprocessed data...")
        data = mne.io.read_raw(os.path.join(store_base_path, f'[preprocessed_prep_asr]p{person_index}.edf'))
    
    recording = eeg_recording.SingleSubjectRecording("0", data)

    #! --- microstate training
    print(f"Begin training microstates. Result will save in '{store_base_path}'")
    print(f" -- Search Microstate Amount from {microstate_search_range[0]} to {microstate_search_range[1]}")
    pre_gev_tot = 0
    for n_states in range(microstate_search_range[0], microstate_search_range[1]):
        print(f"Begin Training {n_states} microstates")
        recording.run_latent_kmeans(n_states = n_states, use_gfp = True, n_inits = n_iters)
        if recording.latent_maps is None:
            continue
        current_gev_tot = recording.gev_tot
        print(f'previous gev_tot = {pre_gev_tot}, current_gev_tot = {current_gev_tot}')
        delta = current_gev_tot - pre_gev_tot
        if delta < stop_delta_threshold:
            break
        if n_states == 4 and store_4_microstates:
            store(recording.latent_maps, recording.latent_segmentation, recording.gev_tot, "[prep-asr]", person_index)
        pre_gev_tot = current_gev_tot
        print(f" -- n_states = {n_states}, gev_tot = {current_gev_tot}. --")
        
    store(recording.latent_maps, recording.latent_segmentation, recording.gev_tot, "[prep-asr]", person_index)

