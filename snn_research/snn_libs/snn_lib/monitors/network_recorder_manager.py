from snn_lib.monitors.recorder import NetworkRecorder
from typing import Dict
class NetworkRecorderManager():
    NEURON_RECORD = 0
    CONNECTION_RECORD = 1
    def __init__(self, neuron_map, connections) -> None:
        self.neuron_map = neuron_map
        self.connections = connections
        self.neuron_recorders : Dict[NetworkRecorder] = {
        }
       
        self.connection_recorders : Dict[NetworkRecorder] = {
        }
       
        self.neuron_recorder_update_states : Dict[bool] = {
            
        }
        
        self.connection_recorder_update_states : Dict[bool]= {
            
        }
        self.requisites = {
        }
    
    def get_neuron_recorder(self, recorder_id):
        return self.neuron_recorders[recorder_id]
    
    def get_connection_recorder(self, recorder_id):
        return self.connection_recorders[recorder_id]
    
    def add_neuron_recorder(self, record_id, item_initializer, update_function):
        self.neuron_recorders[record_id] = NetworkRecorder(item_initializer, update_function)
        self.neuron_recorder_update_states[record_id] = False
        
    def add_connection_recorder(self, record_id, item_initializer, update_function):
        self.connection_recorders[record_id] = NetworkRecorder(item_initializer, update_function)
        self.connection_recorder_update_states[record_id] = False

    def add_pre_requisite(self, record, requisite):
        if record not in self.requisites:
            self.requisites[record] = []
        self.requisites[record].extend(requisite)
        
    def update_all_recorders(self, t, arg = None):
        for recorder_id in self.neuron_recorders:     
            self.update_neuron_recorder(t, recorder_id)
            
        for recorder_id in self.connection_recorders: 
            self.update_connection_recorder(t, recorder_id)
            
        self._finish_update()
            
    def update_neuron_recorder(self, t, recorder_id, arg = None):
        if self.neuron_recorder_update_states[recorder_id] == True:
            return
        if recorder_id in self.requisites:
                prerequisites = self.requisites[recorder_id]
                for prerequisite in prerequisites:
                    if prerequisite[0] == NetworkRecorderManager.NEURON_RECORD:
                        self.update_neuron_recorder(t, prerequisite[1], arg)
                    elif prerequisite[0] == NetworkRecorderManager.CONNECTION_RECORD:
                        self.update_connection_recorder(t, prerequisite[1], arg)
        
        update_function = self.neuron_recorders[recorder_id].update_function

        for neuron_id in self.neuron_map:
            new_value = update_function(t, self, self.neuron_map[neuron_id], neuron_id, self.neuron_recorders[recorder_id].get(neuron_id), arg)
            self.neuron_recorders[recorder_id].update(neuron_id, new_value)
        self.neuron_recorder_update_states[recorder_id] = True
    
    def update_connection_recorder(self, t, recorder_id, arg = None):
        if self.connection_recorder_update_states[recorder_id] == True:
            return
        if recorder_id in self.requisites:
                prerequisites = self.requisites[recorder_id]
                for prerequisite in prerequisites:
                    if prerequisite[0] == NetworkRecorderManager.NEURON_RECORD:
                        self.update_neuron_recorder(t, prerequisite[1], arg)
                    elif prerequisite[0] == NetworkRecorderManager.CONNECTION_RECORD:
                        self.update_connection_recorder(t, prerequisite[1], arg)
        
        update_function = self.connection_recorders[recorder_id].update_function

        for connection in self.connections:
            new_value = update_function(t, self, \
                connection, self.connection_recorders[recorder_id].get(connection[0] + "_" + connection[1]), arg)
            self.connection_recorders[recorder_id].update(connection[0] + "_" + connection[1], new_value)
        self.connection_recorder_update_states[recorder_id] = True
    
    def initialize_recorders(self):
        for recorder_id in self.neuron_recorders:
            self.neuron_recorders[recorder_id].initialize(list(self.neuron_map.keys()))
            
        for recorder_id in self.connection_recorders:
            self.connection_recorders[recorder_id].initialize([connection[0] + "_" + connection[1] for connection in self.connections])

    def _finish_update(self):
        for recorder_id in self.neuron_recorder_update_states:
            self.neuron_recorder_update_states[recorder_id] = False
            
        for recorder_id in self.connection_recorder_update_states:
            self.connection_recorder_update_states[recorder_id] = False
            