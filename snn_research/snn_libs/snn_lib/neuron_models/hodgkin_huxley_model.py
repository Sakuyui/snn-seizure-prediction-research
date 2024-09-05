import numpy as np
from snn_lib.neuron_models.base_neuron_model import AbstractNeuron

class HodgkinHuxleyNeuron(AbstractNeuron):
    def __init__(self, hyperparameters):
        super().__init__()
        ## Hyperparameters
        self.set_hyperparameters(
            {
                'C_m': hyperparameters.get('C_m', 1.0),
                'g_Na': hyperparameters.get('g_Na', 120.0),
                'g_K': hyperparameters.get('g_K', 36.0),
                'g_L': hyperparameters.get('g_L', 0.3),
                'V_Na': hyperparameters.get('V_Na', 50.0),
                'V_K': hyperparameters.get('V_K', -77.0),
                'V_L': hyperparameters.get('V_L', -54.387),
                'dt': hyperparameters.get('dt', 1e-3)
            }
        )

        self.initialize()
        
        self.INDEX_V = 0
        self.INDEX_M = 1
        self.INDEX_H = 2
        self.INDEX_N = 3
        
    def alpha_m(self, V):
        return 0.1 * (V + 40.0)/(1.0 - np.exp(-(V + 40.0) / 10.0))
    def beta_m(self, V):
        return 4.0 * np.exp(-(V + 65.0) / 18.0)
    def alpha_h(self, V):
        return 0.07 * np.exp(-(V + 65.0) / 20.0)
    def beta_h(self, V):
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    def alpha_n(self, V):
        return 0.01 * (V + 55.0)/(1.0 - np.exp(-(V + 55.0) / 10.0))
    def beta_n(self, V):
        return 0.125 * np.exp(-(V + 65) / 80.0)
        
    def initialize(self, maintain_weights = False):
        self._states = [-65, 0.05, 0.6, 0.32]
        self._cached_states = None
        
    @property
    def C_m(self):
        return self._hyperparameters['C_m']
    @property
    def g_Na(self):
        return self._hyperparameters['g_Na']
    @property
    def g_K(self):
        return self._hyperparameters['g_K']
    @property
    def g_L(self):
        return self._hyperparameters['g_L']
    @property
    def V_Na(self):
        return self._hyperparameters['V_Na']
    @property
    def V_K(self):
        return self._hyperparameters['V_K']
    @property
    def V_L(self):
        return self._hyperparameters['V_L']
    @property
    def dt(self):
        return self._hyperparameters['dt']
    
        
    def I_Na(self, V, m, h):
        return self.g_Na * m**3 * h * (V - self.V_Na)
    
    def I_K(self, V, n):
        return self.g_K * n**4 * (V - self.V_K)

    def I_L(self, V):
        return self.g_L * (V - self.V_L)
    
    def pseudo_update_states(self, u):
        states = self.states
        V = states[self.INDEX_V]
        m = states[self.INDEX_M]
        h = states[self.INDEX_H]
        n = states[self.INDEX_N]

        dVdt = (u - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        dmdt = self.alpha_m(V) * (1.0 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1.0 - h) - self.beta_h(V) * h
        dndt = self.alpha_n(V) * (1.0 - n) - self.beta_n(V) * n

        V = V + self.dt * dVdt
        m = m + self.dt * dmdt
        h = h + self.dt * dhdt
        n = n + self.dt * dndt

        self.cache_states([V, m, h, n])
        return self.cached_states
    