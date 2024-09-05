import numpy as np

class ErrorSignal:
    def __init__(self, N_in, N_out, dt=1e-4, td=1e-2, tr=5e-3):
        self.dt = dt
        self.td = td
        self.tr = tr

        self.N_in = N_in
        self.N_out = N_out
        self.r = np.zeros((N_out, N_in))
        self.hr = np.zeros((N_out, N_in))
        self.b = (td / tr) ** (td / (tr - td)) # 規格化定数
        
    def initialize_states(self):
        self.r = np.zeros((self.N_out, self.N_in))
        self.hr = np.zeros((self.N_out, self.N_in))
        
    def __call__(self, output_spike, target_spike):
        r = self.r * (1 - self.dt / self.tr) + self.hr / self.td * self.dt
        hr = self.hr * (1 - self.dt / self.td) + (target_spike - output_spike) /self.b
        self.r = r
        self.hr = hr
        return r

class EligibilityTrace:
    def __init__(self, N_in, N_out, dt=1e-4, td=1e-2, tr=5e-3):
        self.dt = dt
        self.td = td
        self.tr = tr
        self.N_in = N_in
        self.N_out = N_out
        self.r = np.zeros((N_out, N_in))
        self.hr = np.zeros((N_out, N_in))

    def initialize_states(self):
        self.r = np.zeros((self.N_out, self.N_in))
        self.hr = np.zeros((self.N_out, self.N_in))

    def surrogate_derivative_fastsigmoid(self, u, beta = 1, vthr = -50):
        return 1 / (1 + np.abs(beta * (u - vthr))) ** 2
    
    def __call__(self, pre_current, post_voltage):
        # (N_out, 1) x (1, N_in) -> (N_out, N_in)
        pre_ = np.expand_dims(pre_current, axis=0)
        post_ = np.expand_dims(self.surrogate_derivative_fastsigmoid(post_voltage),axis=1)
        r = self.r*(1-self.dt/self.tr) + self.hr*self.dt
        hr = self.hr*(1-self.dt/self.td) + np.dot(post_,pre_) / (self.tr * self.td)
        self.r = r
        self.hr = hr
        return r