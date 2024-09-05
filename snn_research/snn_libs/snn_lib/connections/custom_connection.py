from snn_lib.connections.base_connection import AbstractConnection
from snn_lib.neuron_models.base_neuron_model import AbstractNeuron
import numpy as np

class CustomAllToAllConnection(AbstractConnection):
    def __init__(self, pre_connection_neuron : AbstractNeuron, post_connection_neuron: AbstractNeuron, connection = None, weights = None, mask = None):
        super().__init__()
        self.pre_connection_neuron = pre_connection_neuron
        self.post_connection_neuron = post_connection_neuron
        self.weights = weights      
        if not (mask is None):
            self.mask = mask
        else:
            mask = np.zeros((self.post_connection_neuron.n_neuron, self.pre_connection_neuron.n_neuron))
            mask[connection[0],:] += 1
            mask[:, connection[1]] += 1
            mask[mask != 2] = 0  
            mask[mask == 2] = 1 
            self.mask = mask

    def backward(self, x):
        return self.W * x #self.W.T @ x

    def pseudo_update_states(self, u = None):
        self.cache_states(self.states)
        return self.cached_states

    def get_output(self, u):
        out = np.multiply(self.W, self.mask) * u
        return out #self.W @ u
    
    def initialize(self, W = None, maintain_weights = False):
        size = (self.post_connection_neuron.n_neuron, self.pre_connection_neuron.n_neuron)
        if not maintain_weights:
            if not (self.weights is None):
                W = self.weights
            else:
                W = np.random.rand(size[0], size[1])
            if W.shape[0] != size[0] or W.shape[1] != size[1]:
                raise ValueError("weight shape = [%d, %d] should equal to [%d, %d]" % (W.shape[0], W.shape[1], \
                    size[0], size[1]))
            self.W = W
        self._states = [0]
        self._cached_states = None
        
        

class CustomOneToOneConnection(AbstractConnection):
    def __init__(self, pre_connection_neuron : AbstractNeuron, post_connection_neuron: AbstractNeuron, connection_pre_neuron_indexes, connection_post_neuron_indexes, weights = None, mask = None):
        super().__init__()
        self.pre_connection_neuron = pre_connection_neuron
        self.post_connection_neuron = post_connection_neuron
        self.weights = weights
        
        if len(connection_pre_neuron_indexes) != len(connection_post_neuron_indexes):
            raise ValueError
        
        
        mask = np.zeros((self.post_connection_neuron.n_neuron, self.pre_connection_neuron.n_neuron))
        for neuron_index_pair in zip(connection_pre_neuron_indexes, connection_post_neuron_indexes):
            mask[neuron_index_pair[0], neuron_index_pair[1]] = 1
        self.mask = mask

    def backward(self, x):
        return self.W * x #self.W.T @ x

    def pseudo_update_states(self, u = None):
        self.cache_states(self.states)
        return self.cached_states

    def get_output(self, u):
        out = np.multiply(self.W, self.mask) * u
        return out #self.W @ u
    
    def initialize(self, W = None, maintain_weights = False):
        
        size = (self.post_connection_neuron.n_neuron, self.pre_connection_neuron.n_neuron)
        if not maintain_weights:
            if not (self.weights is None):
                W = self.weights
            else:
                W = np.random.rand(size[0], size[1])
            if W.shape[0] != size[0] or W.shape[1] != size[1]:
                raise ValueError("weight shape = [%d, %d] should equal to [%d, %d]" % (W.shape[0], W.shape[1], \
                    size[0], size[1]))
            self.W = W
        self._states = [0]
        self._cached_states = None
        
        

class CustomConnection(AbstractConnection):
    def __init__(self, pre_connection_neuron : AbstractNeuron, post_connection_neuron: AbstractNeuron, connections, weights = None):
        super().__init__()
        self.pre_connection_neuron = pre_connection_neuron
        self.post_connection_neuron = post_connection_neuron
        self.weights = weights
        
        mask = np.zeros((self.post_connection_neuron.n_neuron, self.pre_connection_neuron.n_neuron))
        for neuron_index_pair in connections:
            mask[neuron_index_pair[0], neuron_index_pair[1]] = 1
        self.mask = mask

    def backward(self, x):
        return self.W * x #self.W.T @ x

    def pseudo_update_states(self, u = None):
        self.cache_states(self.states)
        return self.cached_states

    def get_output(self, u):
        out = np.multiply(self.W, self.mask) * u
        return out #self.W @ u
    
    def initialize(self, W = None, maintain_weights = False):
        size = (self.post_connection_neuron.n_neuron, self.pre_connection_neuron.n_neuron)
        if not maintain_weights:
            if not (self.weights is None):
                W = self.weights
            else:
                W = np.random.rand(size[0], size[1])
            if W.shape[0] != size[0] or W.shape[1] != size[1]:
                raise ValueError("weight shape = [%d, %d] should equal to [%d, %d]" % (W.shape[0], W.shape[1], \
                    size[0], size[1]))
            self.W = W
        self._states = [0]
        self._cached_states = None