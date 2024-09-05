from snn_lib.neuron_models.base_neuron_model import AbstractNeuron
import numpy as np
class IzhikevichNeuronModel(AbstractNeuron):
    def __init__(self, N, hyperparameters):
        super().__init__()
        
        ## Hyperparameters
        self.set_hyperparameters(
            {
                'C': hyperparameters.get('C', 250),
                'a': hyperparameters.get('a', 0.01), 
                'b': hyperparameters.get('b', -2),
                'k': hyperparameters.get('k', 2.5),
                'd': hyperparameters.get('d', 200), 
                'vrest': hyperparameters.get('vrest', -60), 
                'vreset': hyperparameters.get('vreset', -65), 
                'vthr': hyperparameters.get('vthr', -20), 
                'vpeak': hyperparameters.get('vpeak', 30),
                'dt': hyperparameters.get('dt', 0.5)
            }
        )

        ## parameters
        self.neuron_count = N
        
        self.initialize()
        
        self.INDEX_V = 0
        self.INDEX_U = 1
    
    @property
    def C(self):
        return self._hyperparameters['C']
    @property
    def a(self):
        return self._hyperparameters['a']
    @property
    def b(self):
        return self._hyperparameters['b']
    @property
    def k(self):
        return self._hyperparameters['k']
    @property
    def d(self):
        return self._hyperparameters['d']
    @property
    def vrest(self):
        return self._hyperparameters['vrest']
    @property
    def vreset(self):
        return self._hyperparameters['vreset']
    @property
    def vthr(self):
        return self._hyperparameters['vthr']
    @property
    def vpeak(self):
        return self._hyperparameters['vpeak']
    @property
    def dt(self):
        return self._hyperparameters['dt']
    
    def initialize_states(self):
        self._states = [self.vrest * np.ones(self.n_neuron), np.zeros(self.n_neuron)]
        self._cached_states = None
        
    def pseudo_update_states(self, I):
        states = self.states
        v = states[self.INDEX_V]
        u = states[self.INDEX_U]
        
        dv = (self.k * (v - self.vrest) * (v - self.vthr) - u + I) / self.C
        v_new = v + self.dt * dv
        u_new = u + self.dt * (self.a * (self.b * (v - self.vrest) - u))
        s = 1* (v >= self.vpeak) 

        # reset when arrive threshold.
        v_new = v_new * (1 - s) + self.vreset * s
        u_new = u_new + self.d * s

        
        self.cache_states([v, u])
        return self.cached_states
   