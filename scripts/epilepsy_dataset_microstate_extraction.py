import numpy as np
import mne
import os, sys
from datetime import datetime


sys.path.append("../data/dataset")
sys.path.append("..//microstate_lib/code")
from dataset import *
import eeg_recording
import asrpy

dataset_base_path = "../data"
dataset_facade = EEGDatasetFacade(dataset_base_path=dataset_base_path)
dataset = dataset_facade("epileptic_eeg_dataset")

record_indexes = {
    # '10': [[10, 1], [10, 2]], 
    '11': [[11, 1], [11, 2], [11, 3], [11, 4]], 
    '12': [[12, 1], [12, 2], [12, 3]],
    '13': [[13, 1], [13, 2], [13, 3], [13, 4]], 
    '14': [[14, 1], [14, 2], [14, 3]],
    '15': [[15, 1], [15, 2], [15, 3], [15, 4]]
}

microstate_search_range = (4, 16)
n_iters = 100
stop_delta_threshold = 0.025
store_4_microstates = True
save_preprocessed_data = True

store_base_path = dataset.base_path
cut_off = 30

global start_time
global end_time
start_time = -1
end_time = -1
save_segmentation = True

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

load_preprocessing = set(['10'])
    
for person_index in record_indexes:
    print(f"Train microstates for person {person_index}")
    record_index_list = record_indexes[person_index]
    
    pre_gev_tot = 0
    if person_index not in load_preprocessing:
        data_count = len(record_index_list)
        ast_results = []
        for slice_begin in range(0, data_count, 2):
            data = dataset.get_merge_mne_data(record_index_list)
            #! --- preprocessing ---
            print(f"[Preprocessing 1: {slice_begin // 2 + 1}/{int(np.ceil(data_count / 2))}]... ASR, cutoff = {cut_off}")
            asr = asrpy.ASR(sfreq=data.info["sfreq"], cutoff=cut_off)

            data.load_data()
            asr.fit(data)
            data = asr.transform(data)
            ast_results.append(data)
        data = mne.concatenate_raws(ast_results)
        del ast_results
        print("[Preprocessing 2]... reference to average")
        data = data.set_eeg_reference('average')
        if save_preprocessed_data:
            mne.export.export_raw(os.path.join(store_base_path, f'[preprocessed]person{person_index}.edf'), data)
    else:
        print(f"Load preprocessed data...")
        data = mne.io.read_raw(os.path.join(store_base_path, f'[preprocessed]person{person_index}.edf'))
    
    recording = eeg_recording.SingleSubjectRecording("0", data)

    #! --- microstate training
    print(f"Begin training microstates. Result will save in '{store_base_path}'")
    print(f" -- Search Microstate Amount from {microstate_search_range[0]} to {microstate_search_range[1]}")

    for n_states in range(microstate_search_range[0], microstate_search_range[1]):
        print(f"Begin Training {n_states} microstates")
        recording.run_latent_kmeans(n_states = n_states, use_gfp = True, n_inits = n_iters)
        current_gev_tot = recording.gev_tot
        delta = current_gev_tot - pre_gev_tot
        if delta < stop_delta_threshold:
            break
        if n_states == 4 and store_4_microstates:
            store(recording.latent_maps, recording.latent_segmentation, recording.gev_tot, "[avg+asr]", person_index)
        pre_gev_tot = current_gev_tot
    store(recording.latent_maps, recording.latent_segmentation, recording.gev_tot, "[avg+asr]", person_index)