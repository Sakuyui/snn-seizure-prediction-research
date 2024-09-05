from snn_lib.neuron_models.base_neuron_model import AbstractNeuron
from typing import Callable, List
from numpy.typing import NDArray
import numpy as np

class FixedSpikeTrainNeuronModel(AbstractNeuron):
 
    def __init__(self, N, spikes, hyperparameters):
        super().__init__()

        ## Hyperparameters
        self.set_hyperparameters({
                'dt': hyperparameters.get('dt', 1e-4), 
                'simulation_time_duration': hyperparameters.get('simulation_time_duration', 1),
            })

        ## parameters
        self.time_steps = (int)(np.ceil(self.simulation_time_duration / self.dt))
        self.spikes = spikes
        if N != spikes.shape[1]:
            raise ValueError("N = %d should equal to the spikes's 2-nd dim, which has size of %d" % (N, spikes.shape[1]))
        self.neuron_count = N
        self.INDEX_SPIKE_OUT = 0
        self.INDEX_T = 1
        self.INDEX_V = 2

        self.initialize()

    def _generate_spikes(self, spikes = None):
        if spikes == None:
            return
        self.spikes = spikes
        
    @property
    def dt(self):
        return self._hyperparameters['dt']
    
    @property
    def simulation_time_duration(self):
        return self._hyperparameters['simulation_time_duration']

    def reset_spikes(self, spikes):
        self.spikes = spikes
        
    def pseudo_update_states(self, u = None):
        t = self.states[self.INDEX_T]
        neurons_v = self.spikes[t]
        t += 1

        self.cache_states([neurons_v, t, 0])
        
        return self.cached_states
    
    def initialize(self):
        self._states = [self.spikes[0], 0, 0]
        self._cached_states = None
        
        
