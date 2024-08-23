import pandas as pd
import numpy as np
import os, sys

class FeatureRecorder():
    def __init__(self, recorder_config) -> None:
        self.logs = []
        self.records = {}
        self.length = 0
        self.config = recorder_config
        self.temporary_variables = {}
        for config_item in recorder_config:
            self.records[config_item] = []
        self.temporary_variables = {}

    def begin_recording(self,  remain_content=True, remain_record_objective=True):
        if not remain_record_objective:
            self.clear_all()
        elif not remain_content:
            self.clear_all_content()

        self.logs.append([self.length, -1])

    def end_recording(self):
        self.logs[-1][1] = self.length - 1
            
    def record_at_t(self, window, time = -1):
        #   self.record_objects[name] = {'enabled': True, 
        #                                     'func': lambda win, **dep_kwargs: func(**({'win': win} | inject_arguments | dep_kwargs)), 
        #                                     'dependencies': [], 'temporary': temporary, 'apply_per_channel':apply_per_channel}
    
        record_name = 0
        def _record():
            if time < 0 or time >= self.length:
                self.records[record_name].append(val)
            else:
                self.records[record_name][time] = val
        def _record_temporary():
            self.temporary_variables[record_name] = val
            
        for record_name in self.records:
            item = self.config[record_name]
            if not item['enabled']:
                val = None
            else:
                func = item['func']
                if item['apply_per_channel']:
                    val = [func(window[:, channel_id], **{item['inject_arguments']}) for channel_id in range(window.shape[1])]
                else:
                    val = func(window, **{item['inject_arguments']})
            if item['temporary']:
                _record_temporary()
            else:
                _record()
        self.length += 1
            
    def append_record(self, window = -1):
        self.record_at_t(window, -1)
        
    def clear_all_content(self):
        return
    
    def clear_all():
        return
    def get_dataframe(self):
        return pd.DataFrame(self.records)
    
class RecordConfiguration():
    def __init__(self) -> None:
        self.record_objects = {
        }

        
    def add_record_object(self, name, func, dependencies = [], temporary = False, inject_padding = None, inject_arguments = {}, apply_per_channel=False, inject_recorder=False):
        self.record_objects[name] = {'enabled': True, 
                                            'func': lambda win, **ext_kwargs: func(**({'win': win} | inject_arguments | ext_kwargs)), 
                                            'dependencies': dependencies, 'temporary': temporary, 'apply_per_channel':apply_per_channel, "inject_padding":inject_padding,
                                            'inject_recorder': inject_recorder}

    
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
        self.initialized = False
        
    def initilization(self, recorder_configuration, append_record_mode=True):
        self._calculate_recording_path(recorder_configuration)
        self.initialized = True
        
    def _calculate_recording_path(self, recorder_configuration):
        key_list = list(recorder_configuration)
        configuration_id_map = {key:index for index, key in enumerate(key_list)}
        n = len(configuration_id_map)
        dependency_graph = np.zeros((n, n))
        for i in range(n):
            dependencies = recorder_configuration[key_list[i]]['dependencies']
            for dep in dependencies:
                dependency_graph[configuration_id_map[dep], i] = 1
        path = []
        while len(path) < n:
            processed = False
            for i in range(n):
                if all(x == 0 for x in dependency_graph[:, i]):
                    if all(x == 0 for x in dependency_graph[i, :]):
                        continue
                    dependency_graph[i, :][dependency_graph[i, :] == 1] = 0
                    processed = True
                    path.append(key_list[i])
            if not processed:
                raise ArithmeticError
        self.path = path

    def _reset_recorder(self, feature_recorder, append_record_mode=True):
        if append_record_mode:
            feature_recorder.begin_recording(True, True)
        else:
            feature_recorder.begin_recording(False, True)
            
    def do_extraction(self, signal, feature_recorder, time_limit = -1):
        current_time = 0
        def window_views(signal, time_delay):
            T = signal.shape[0]
            for t in range(0, T):
                yield signal[max(0, t - time_delay): t + 1, :]
        
        feature_recorder.begin_recording()

        for window in window_views(signal, self.freq * self.decision_time_delay):
            feature_recorder.record_at_t(current_time, window)

            current_time += 1
            if time_limit >= 0 and current_time >= time_limit:
                break
            
        feature_recorder.end_recording()
        return feature_recorder