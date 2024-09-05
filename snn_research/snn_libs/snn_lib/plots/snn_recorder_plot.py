import matplotlib.pyplot as plt
import numpy as np
from snn_lib.monitors.network_recorder_manager import NetworkRecorderManager
def plot_neuron_record(recorder, key, title_generator, xlabel="", ylabel="", dt=1, figsize = (25, 45)):
    fig = plt.figure(figsize=figsize)
    rows = int(np.ceil(sum([recorder.neuron_map[n].n_neuron for n in recorder.neuron_map]) / 3))
    cols = 3
    current_index = 0
    iterations = recorder.neuron_recorders[key].record
    for record_id in iterations:
        record = iterations[record_id]
        items = (np.array([list(r) if hasattr(r, '__iter__') else [r] for r in record]))
        time_steps = items.shape[0]
        for i in range(items.shape[1]):
            plt.subplot(rows, cols, current_index + 1)
            plt.tight_layout() 
            plt.plot(np.arange(int(time_steps)) * dt, items[:, i] , color="k")
            plt.xlim(0, time_steps * dt)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            
            # 'Neuron-id = "' + record_id + ('_%d' % i) + '"\'s  Membrane Potential.'
            plt.title(title_generator(record_id, i), fontsize = 18)
            current_index += 1

