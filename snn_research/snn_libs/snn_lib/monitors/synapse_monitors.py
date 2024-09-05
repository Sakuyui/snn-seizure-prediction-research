import numpy
class SynapseOutputMonitor(object):
    def __init__(self):
        self.synapse_output_records = {}

    def record_synapse_output(self, t, synapse, synapse_id):
        if synapse_id not in self.synapse_output_records:
            self.synapse_output_records[synapse_id] = {}
        out = synapse.states[synapse._output_index]
        if isinstance(out, (list, numpy.ndarray)):
            if len(out) == 1:        
                out = out[0]
        self.synapse_output_records[synapse_id][t] = out
        
    def clear_record(self):
        self.synapse_output_records = []
        
    def get_dataframe_record(self):
        import pandas as pd
        return pd.DataFrame(self.synapse_output_records)