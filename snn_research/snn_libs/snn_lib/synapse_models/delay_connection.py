import numpy as np
from snn_lib.synapse_models.base_synapse import AbstractSynapse

class DelayConnection(AbstractSynapse):
    def __init__(self, N, delay, dt=1e-4):
        super().__init__()
        nt_delay = round(delay / dt)
        self.N = N
        self.delay = delay
        self.dt = dt
        self.nt_delay = nt_delay
        self.initialize()
    
    def initialize(self):
        self._states = [0, np.zeros((self.N, self.nt_delay))]
        self._cached_states = None
    
    
    def pseudo_update_states(self, u=None):
        states = self.states[1]
        states[:, 1:] = states[:, :-1]
        states[:, 0] = u
        out = states[:, -1]
        self._cached_states = [out, states]
        
        return self._cached_states