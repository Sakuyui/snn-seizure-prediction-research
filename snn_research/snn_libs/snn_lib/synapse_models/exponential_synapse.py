import numpy as np
from snn_lib.synapse_models.base_synapse import AbstractSynapse

class SingleExponentialSynapse(AbstractSynapse):
    def __init__(self, N, dt=1e-4, td=5e-3):
        super().__init__()
        self.N = N
        self.dt = dt
        self.td = td
        self.INDEX_R = 0
    
    def initialize(self):
        self._states = [np.zeros(self.N)]
        self._cached_states = None
               
    
    def pseudo_update_states(self, u=None):
        r = self._cached_states[self.INDEX_R]
        r = r * (1 - self.dt / self.td) + u / self.td
        self._cached_states[self.INDEX_R] = r
    
class DoubleExponentialSynapse(AbstractSynapse):
    def __init__(self, pre_N, post_N, dt=1e-4, td=1e-2, tr=5e-3):
        super().__init__()
        self.pre_N = pre_N
        self.post_N = post_N
        self.dt = dt
        self.td = td
        self.tr = tr
        self.INDEX_R = 0
        self.INDEX_HR = 1
        
    def initialize(self):
        self._states = [np.zeros((self.post_N, self.pre_N)), np.zeros((self.post_N, self.pre_N))]
        self._cached_states = None

    def pseudo_update_states(self, u = None):
        r = self.states[self.INDEX_R] * (1 - self.dt / self.tr) + self.states[self.INDEX_HR] * self.dt
        hr = self.states[self.INDEX_HR] * (1 - self.dt / self.td) + u / (self.tr * self.td)
        self._cached_states = [r, hr]
        return self._cached_states
    