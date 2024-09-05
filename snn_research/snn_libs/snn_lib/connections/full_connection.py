from snn_lib.connections.base_connection import AbstractConnection
import numpy as np

class FullConnection(AbstractConnection):
    def __init__(self, N_in, N_out, initW = None):
        super().__init__()
        self.N_in = N_in
        self.N_out = N_out
        if initW is not None:
            self.initialize(initW)
        else:
            self.initialize()

    def backward(self, x):
        return np.dot(self.W.T, x) #self.W.T @ u

    def pseudo_update_states(self, u=None):
        self.cache_states(self.states)
        return self.cached_states

    def get_output(self, u):
        return np.dot(self.W, u) #self.W @ u
    
    def initialize(self, W = None, maintain_weights = False):
        if not maintain_weights == False:
            if W == None:
                self.W = 0.1 * np.random.rand(self.N_out, self.N_in)
            else:
                self.W = W
        self._cached_states = None
            
