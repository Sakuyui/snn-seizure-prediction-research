from snn_lib.monitors.neuron_monitors import NeuronOutputMonitor, NeuronMembranePotentialMonitor
from snn_lib.monitors.synapse_monitors import SynapseOutputMonitor
import numpy as np

class Network():
    def __init__(self, neurons, connections, enable_monitors = True) -> None:
        self.neurons = neurons
        self.connections = connections
        self.enable_monitors = enable_monitors
        if enable_monitors:
            self.neuron_output_monitor = NeuronOutputMonitor()
            self.neuron_membrane_potential_monitor = NeuronMembranePotentialMonitor()
            self.synapse_output_monitor = SynapseOutputMonitor()


    def initialize_network(self, maintain_weights = False):
        for neuron_id in self.neurons:
            neuron = self.neurons[neuron_id]
            neuron.initialize()
            neuron.reset_time()
            if self.enable_monitors:
                self.neuron_output_monitor.record_neuron_output(0, neuron, neuron_id)
                self.neuron_membrane_potential_monitor.record_neuron_membrane_potential(0, neuron, neuron_id)

        for connection in self.connections:
            connection[2].initialize(maintain_weights = maintain_weights)
            connection[3].initialize()
            synapse_id = f's[{connection[0]}_{connection[1]}]'
            if self.enable_monitors:
                self.synapse_output_monitor.record_synapse_output(0, connection[3], synapse_id)
            
        self.neuron_in_connection_map = {
            
        }
        for neuron_id in self.neurons:
            self.neuron_in_connection_map[neuron_id] = []
            for connection in self.connections:
                if connection[1] == neuron_id:
                    self.neuron_in_connection_map[neuron_id].append(connection)
    
    
    def forward_single_time_step(self, t, train_recorder):
        def evolve_neuron(neuron_id):
            neuron= self.neurons[neuron_id]
            connections_to_current_neuron = self.neuron_in_connection_map[neuron_id]
            
            total_synapse_out = np.zeros((neuron.n_neuron))
        
            for connection in connections_to_current_neuron:
                synapse = connection[3]
                synapse_states, synapse_out = synapse.states, synapse.states[synapse._output_index]
                total_synapse_out += synapse_out.sum(axis = 1)
                
      
            return neuron(total_synapse_out)
            
        for neuron_id in self.neurons:
            evolve_neuron(neuron_id)

        for connection in self.connections:
            conn = connection[2]
            synapse = connection[3]
            pre_neuron = self.neurons[connection[0]]
            post_neuron = self.neurons[connection[1]]
            connection_states, connection_output = conn(pre_neuron.states[pre_neuron._output_index])

            synapse_states, synapse_output = synapse(connection_output)

            
        # update states of neurons and synapses. (states x_{t} evolve to x_{t + 1})
        for neuron_id in self.neurons:
            neuron = self.neurons[neuron_id]
            neuron.do_update_states()
            if self.enable_monitors:
                self.neuron_output_monitor.record_neuron_output(t + 1, neuron, neuron_id)
                self.neuron_membrane_potential_monitor.record_neuron_membrane_potential(t + 1, neuron, neuron_id)
            
        for connection in self.connections:
            connection[2].do_update_states()
            connection[3].do_update_states()
            synapse_id = f's[{connection[0]}_{connection[1]}]'

            # record and monitor
            # connection_history_weights[connection[0] + '_' + connection[1]].append(connection[2].W)
            if self.enable_monitors:
                self.synapse_output_monitor.record_synapse_output(t + 1, connection[3], synapse_id)

        train_recorder.update_all_recorders(t)