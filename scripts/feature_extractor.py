import pandas as pd
import numpy as np
import os, sys


class FeatureRecorder():
    def __init__(self, recorder_config) -> None:
        self.logs = []
        self.records = {}
        self.length = 0
        self.config = recorder_config
        for config_item in recorder_config:
            self.records[config_item] = []

    def begin_recording(self,  remain_content=True, remain_record_objective=True):
        if not remain_record_objective:
            self.clear_all()
        elif not remain_content:
            self.clear_all_content()
        self.logs.append([self.length, -1])

    def end_recording(self):
        self.logs[-1][1] = self.length - 1
            
    def record_at_t(self, window, time = -1):
        record_name = 0
        for record_name in self.records:
            item = self.config[record_name]
            if not item['enabled']:
                val = None
            else:
                func = item['func']
                val = func(window)
            if time < 0 or time >= self.length:
                self.records[record_name].append(val)
            else:
                self.records[record_name][time] = val
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
        self.record_objectives = {
        }
    def add_record_object(self, name, func, dependencies = [], temporary = False):
        self.record_objectives[name] = {'enabled': True, 'func': func, 'dependencies': [], 'temporary': temporary}

    def enable_record_item(self, name):
        if name not in self.record_objectives:
            return
        self.record_objectives[name]['enabled'] = True
        
    def __iter__(self):
        return self.record_objectives.__iter__()
    
    def __getitem__(self, key):
        return self.record_objectives[key]
        
class FeatureExtractor():
    def __init__(self, freq, decision_time_delay):
        self.freq = freq
        self.decision_time_delay = decision_time_delay
        self.initialized = False
        
    def initilization(self, recorder_configuration, append_record_mode=True):
        self._calculate_recording_path(recorder_configuration)
        self.initialized = True
        
    def _calculate_recording_path(self, recorder_configuration):
        path = []
        white = list(recorder_configuration)
        grey = []
        
        while white or len(grey) > 0:
            if len(grey) == 0:
                key = white.pop()
                grey.append(key)
            else:
                dependencies = recorder_configuration[grey[-1]]['dependencies']
                all_black = True
                for dep in dependencies:
                    if dep in grey:
                        all_black = False
                        continue
                    if dep in path:
                        continue
                    
                    grey.append(dep)
                    white.remove(dep)
                    
                if all_black:
                    key_record_item_can_be_processed = grey.pop()
                    path.append(key_record_item_can_be_processed)
        self.path = path

    def _reset_recorder(self, feature_recorder, append_record_mode=True):
        if append_record_mode:
            feature_recorder.begin_recording(True, True)
        else:
            feature_recorder.begin_recording(False, True)
            
    def do_extraction(self, signal, feature_recorder, append_record_mode = True, time_limit = -1):
        current_time = 0
        def window_views(signal, time_delay):
            T = signal.shape[0]
            for t in range(0, T):
                yield signal[max(0, t - time_delay): t + 1, :]
        
        self._reset_recorder(feature_recorder, append_record_mode)

        for window in window_views(signal, int(self.freq * self.decision_time_delay)):
            if append_record_mode:
                feature_recorder.append_record(window)
            else:
                feature_recorder.record_at_t(window, current_time)
            current_time += 1
            if time_limit >= 0 and current_time >= time_limit:
                break
            
        feature_recorder.end_recording()
        return feature_recorder