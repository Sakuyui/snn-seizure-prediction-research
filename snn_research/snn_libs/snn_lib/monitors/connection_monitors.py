import numpy
def ConnectionWeightsMonitor():
   
    def __init__(self):
        self.connection_weight_history = {}

    def record_neuron_output(self, t, connection, connection_id):
        if connection_id not in self.connection_weight_history:
            self.connection_weight_history[connection_id] = {}
        out = connection.W
        if isinstance(out, (list, numpy.ndarray)):
            if len(out) > 1:
                raise ValueError
            out = out[0]
        self.connection_weight_history[connection_id][t] = out
        
    def clear_record(self):
        self.connection_weight_history = {}
        
    def get_dataframe_record(self):
        import pandas as pd
        return pd.DataFrame(self.connection_id)
    
    def get_data_at_time_t(self, t):
        return {connection_id: self.connection_weight_history[connection_id][t] for connection_id in self.connection_weight_history}
    
    def get_data_at_time_t_vector_form(self, t):
        return [self.connection_weight_history[connection_id][t] for connection_id in self.connection_weight_history]