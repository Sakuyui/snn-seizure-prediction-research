import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sklearn.decomposition import IncrementalPCA
import time as time_module
from joblib import Parallel, delayed  

def create_shared_memory_nparray(data, np_data_type, shape, name):
    d_size = np.dtype(np_data_type).itemsize * np.prod(shape)

    shm = shared_memory.SharedMemory(create=True, size=d_size, name=name)
    # numpy array on shared memory buffer
    dst = np.ndarray(shape=shape, dtype=np_data_type, buffer=shm.buf)
    dst[:] = data[:]
    print(f'NP SIZE: {(dst.nbytes / 1024) / 1024}')
    return shm

class FunctionWrapper:
    def __init__(self, func, inject_arguments):
        self.func = func
        self.inject_arguments = inject_arguments

    def __call__(self, win, ext_kwargs):
        return self._call_func(self.func, win, ext_kwargs, self.inject_arguments)

    def _call_func(self, func, win, ext_kwargs, inject_arguments):
        # Assuming this is the logic from your original _call_func
        return func(win, **{**ext_kwargs, **inject_arguments})
    
class FeatureRecorder():
    def __init__(self, recorder_config, record_path = None) -> None:
        self.logs = []
        self.records = {}
        self.ipca = None
        self.length = 0
        self.config = recorder_config
        self.temporary_variables = {}
        for config_item in recorder_config:
            self.records[config_item] = []
        self.temporary_variables = {}
        self.record_path = record_path
        self.prepared = False
        self._profile_performance = False
        self.enable_pca = False
        self.profiled_performance = {}
    
    def use_pca(self, n_components = 4):
        self.ipca = IncrementalPCA(n_components=n_components)
        self.enable_pca = True
        return self
    
    def disable_pca(self):
        self.enable_pca = False

    def save_all(self, path, format = "%s.npy"):
        
        for name in self.records.keys():
            config = self.config[name]
            if config['temporary']:
                continue
            np.save(os.path.join(path, format % name), np.array(self.records[name], dtype=object), allow_pickle=True)
    
    def begin_recording(self,  remain_content=True, remain_record_objective=True, profile_performance = False):
        if not remain_record_objective:
            self.clear_all()
        elif not remain_content:
            self.clear_all_content()

        self.logs.append([self.length, -1])
        self._profile_performance = profile_performance

    def end_recording(self):
        self.logs[-1][1] = self.length - 1
    
    def eval_item(self, record_name, src_signal, src_window, pca_windows, win_begin, time):
            if self._profile_performance:
                start_t = time_module.time()
            item = self.config[record_name]
            if item['use_pca']:
                window = pca_windows
            else:
                window = src_window
                
            if not item['enabled']:
                val = None
            else:
                extend_args = {}
                func = item['func']
                if 'inject_padding' in item:
                    padding = item['inject_padding']
                    if 'padding_left' in padding:
                        padding_fragment = src_signal[win_begin - padding['padding_left'] : win_begin]
                        if self.enable_pca and item['use_pca']:
                            padding_fragment = self.ipca.transform(padding_fragment)
                        extend_args |= {'padding_left': padding_fragment}
                    if 'padding_right' in padding:
                        padding_fragment = src_signal[win_begin + window.shape[0] : win_begin + window.shape[0] + padding['padding_right']]
                        if self.enable_pca and item['use_pca']:
                            padding_fragment = self.ipca.transform(padding_fragment)
                        extend_args |= {'padding_right': padding_fragment}
                    
                for dep in item['dependencies']:
                    extend_args |= {dep: self.records[dep][-1]}
                    
                if item['apply_per_channel']:
                    val = [func(window[:, channel_id], extend_args | {'channel_id': channel_id}) for channel_id in range(window.shape[1])]
                else:
                    val = func(window, extend_args)
                    
            if item['temporary']:
                self._record_temporary(record_name, val)
            else:
                self._record(record_name, val, time=time)
                
            if self._profile_performance:
                self.profiled_performance[record_name] = time_module.time() - start_t
    
    def _record(self, record_name, val, time=-1):
            if time < 0 or time >= self.length:
                self.records[record_name].append(val)
            else:
                self.records[record_name][time] = val
    
    def _record_temporary(self, record_name, val):
            self.records[record_name] = [val]
    
    def eval_parallelizable_block(self, block, src_signal, src_window, pca_windows, win_begin, time):
        Parallel(n_jobs=1, backend="multiprocessing")(delayed(self.eval_item)(record_name, src_signal, src_window, pca_windows, win_begin, time) for record_name in block)
        
    def record_at_t(self, src_signal, win_begin, window, time = -1, parallel=False):
        record_name = 0
        src_window = window
        if self.enable_pca:
            pca_windows = self.ipca.fit_transform(window)
            
        if self._profile_performance:
            self.profiled_performance = {}
        
        if not parallel:
            for record_name in self.record_path:
                self.eval_item(record_name=record_name, src_signal=src_signal, src_window = src_window, pca_windows = pca_windows, win_begin=win_begin, time=time)
        else:
            for block in self.record_path:
                self.eval_parallelizable_block(block=block, src_signal=src_signal, src_window = src_window, pca_windows = pca_windows, win_begin=win_begin, time=time)
                
        self.length += 1
        if self._profile_performance:
            return self.profiled_performance
            
    def append_record(self, src_signal, win_begin, window = -1):
        self.record_at_t(src_signal, win_begin, window, -1)
        
    def clear_all_content(self):
        return
    
    def clear_all():
        return
    
    def get_dataframe(self):
        return pd.DataFrame(self.records)
    
class RecordConfiguration():
    def __init__(self, record_objects = {}) -> None:
        self.record_objects = record_objects
        
    def _call_func(self, func, win, ext_kwargs, inject_arguments):
        total_arguments = {'win': win} | inject_arguments | ext_kwargs
        return func(**total_arguments)
        
    def lambda_wrapper(self, func, inject_arguments):
        return lambda win, ext_kwargs: self._call_func(func=func, win=win, ext_kwargs=ext_kwargs, inject_arguments=inject_arguments)
    
    def add_record_object(self, name, func, dependencies = [], temporary = False, inject_padding = {}, inject_arguments = {}, apply_per_channel=False, inject_recorder=False, use_pca=False):
        self.record_objects[name] = {'enabled': True, 
            'func': FunctionWrapper(func=func, inject_arguments=inject_arguments), 
            'dependencies': dependencies, 'temporary': temporary, 'apply_per_channel':apply_per_channel, "inject_padding":inject_padding,
            'inject_recorder': inject_recorder, 'use_pca': use_pca}
        
    def enable_record_item(self, name):
        if name not in self.record_objects:
            return
        self.record_objects[name]['enabled'] = True
        
    def __iter__(self):
        return self.record_objects.__iter__()
    
    def __getitem__(self, key):
        return self.record_objects[key]
        
class FeatureExtractor():
    def __init__(self, freq, decision_time_delay):
        self.freq = freq
        self.decision_time_delay = decision_time_delay
        
    def profile_performance(self, value):
        self._profile_performance = value
        
    def _reset_recorder(self, feature_recorder, append_record_mode=True):
        if append_record_mode:
            feature_recorder.begin_recording(True, True, self._profile_performance)
        else:
            feature_recorder.begin_recording(False, True, self._profile_performance)
            
    def do_extraction(self, signal, feature_recorder : FeatureRecorder, time_limit = -1, parallel=False) -> FeatureRecorder:
        if not feature_recorder.prepared:
            raise ArithmeticError("Feature Recorder should be built before performing extraction.")
        current_time = 0
            
        def window_views(signal, time_delay):
            T = signal.shape[0]
            for t in range(time_delay, T):
                # print(f'windows from {max(0, t - time_delay)} to {t + 1}', signal[max(0, t - time_delay): t + 1, :].shape)
                yield signal[max(0, t - time_delay): t + 1, :]
        feature_recorder.begin_recording(profile_performance=self._profile_performance)

        total_iterations =  signal.shape[0] - self.decision_time_delay
        performance = {}
        
        for window in tqdm(window_views(signal, self.decision_time_delay), total = total_iterations,  desc="Extracting features"):
            if self._profile_performance:
                performance_one_iter = feature_recorder.record_at_t(src_signal=signal, win_begin=self.decision_time_delay + current_time, window=window, time=current_time, parallel=parallel)
                performance = {key: performance.get(key, 0.0) + performance_one_iter[key] for key in performance_one_iter}
            else:
                feature_recorder.record_at_t(signal, self.decision_time_delay + current_time * self.decision_time_delay, window, current_time, parallel=parallel)
            
            current_time += 1
            if time_limit >= 0 and current_time >= time_limit:
                break
            
        feature_recorder.end_recording()
        if self._profile_performance:
            return feature_recorder, performance
        return feature_recorder