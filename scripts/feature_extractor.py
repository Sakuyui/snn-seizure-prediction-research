freq = 500
decision_time_delay = freq * 10

# sliding_window
class FeatureRecorder():
    def __init__(self) -> None:
        self.logs = []
        self.records = {}
        self.length = 0
    
    def begin_recording(self, recorder_config,  remain_content=True, remain_record_objective=True):
        if not remain_record_objective:
            feature_recorder.clear_all()
        elif not remain_content:
            feature_recorder.clear_all_content()
        
        self.logs.append([self.length, -1])
    
    def end_recording(self):
        self.logs[-1][1] = self.length - 1
            
    def record_at_t(self, time, window):
        pass
    
    def clear_all_content(self):
        return
    def clear_all():
        return
    
class RecordConfiguration():
    def __init__(self) -> None:
        self.record_objectives = {
        }
    def add_record_object(self, name, func):
        self.record_objectives[name] = func
        
    
    

class FeatureExtractor():
    def __init__(self, freq, decision_time_delay):
        self.freq = freq
        self.decision_time_delay = decision_time_delay
        
    def setup_recorder(self,  feature_recorder, recorder_configuration, remain_content=False, remain_record_objective=False):
        if not remain_record_objective:
            feature_recorder.clear_all()
            
        elif not remain_content:
            feature_recorder.clear_all_content()
        
        for config_key in recorder_configuration:
            feature_recorder.add_record_object(config_key, recorder_configuration[config_key])
    
    def do_extraction(self, signal, recorder):
        current_time = 0
        feature_recorder = FeatureRecorder()
        def window_views(signal, time_delay):
            T = signal.shape[0]
            for t in range(0, T):
                yield signal[max(0, t - time_delay): t, :]
        
        feature_recorder.begin_recording(recorder)
            
            
        for window in window_views(signal, self.freq * self.decision_time_delay):
            feature_recorder.record_at_t(current_time, window)
            current_time += 1
            
        feature_recorder.end_recording()
        return feature_recorder