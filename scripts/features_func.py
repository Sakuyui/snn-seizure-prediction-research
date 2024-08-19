import numpy as np
def flatten_remove_diag(x):
    x_no_diag = np.ndarray.flatten(x)
    x_no_diag = np.delete(x_no_diag, range(0, len(x_no_diag), len(x) + 1), 0)
    x_no_diag = x_no_diag.reshape(len(x), len(x) - 1)
    return x_no_diag.ravel()

def electrode_correlation(win, n_channels):
    return [[0]] * (n_channels * (n_channels - 1) // 2) if win.shape[0] < 2 else np.corrcoef(win.T)[np.triu_indices(n_channels, k = 1)]

def calculate_auto_corr(signal, lag):
    corr = 0
    for t in range(lag, signal.shape[0]):
        corr += np.multiply(signal[t - lag, :], signal[t, :])
    return corr

def distribution_entropy(win, time_step_lag = 15):
    mut_inf = mutual_information(win, time_step_lag)
    return np.sum([p_ij * np.log(p_ij) for p_ij in mut_inf / np.sum(mut_inf)])

def network_entropy(win, time_step_lag=15):
    mutual_information_2d = -0.5 * (1 - (np.corrcoef(win.T, np.roll(win.T, -time_step_lag)))**2)
    np.fill_diagonal(mutual_information_2d, 0)
    s = np.repeat(np.sum(mutual_information_2d, axis = 0), win.shape[1] - 1)
    return flatten_remove_diag(mutual_information_2d) / s

def mutual_information(win, time_step_lag=15):
    mutual_information = flatten_remove_diag(-0.5 * (1 - (np.corrcoef(win.T, np.roll(win.T, -time_step_lag)))**2))
    return mutual_information

def visuality_graph(one_dimensional_signal):
    width = len(one_dimensional_signal)
    value_diff = np.repeat(one_dimensional_signal[:,np.newaxis].T, width, axis=0) - np.repeat(one_dimensional_signal[:,np.newaxis], width, axis=1)
    time_differ = np.repeat(np.arange(0, width)[:,np.newaxis].T, width, axis=0) - np.repeat(np.arange(0, width)[:,np.newaxis], width, axis=1)
    graph = np.arctan(value_diff / value_diff)
    return graph

def conditional_transfer_information(win, time_lags):
    raise NotImplementedError

def electrode_local_entropy(win):
    raise NotImplementedError

def vg_graph_index_complexity(win):
    raise NotImplementedError

def vg_degree_distribution(win):
    raise NotImplementedError

def vg_degree_entropy(win):
    raise NotImplementedError

def vg_degree_centrality(win):
    raise NotImplementedError

def vg_jaccard_similarity_coefficient(win):
    raise NotImplementedError

def vg_modularity(win):
    raise NotImplementedError

def vg_average_shortest_path_length(win):
    raise NotImplementedError

def vg_closeness_centrality(win):
    raise NotImplementedError

def vg_small_world(win):
    raise NotImplementedError

def vg_density(win):
    raise NotImplementedError

def vg_eigenvector_centrality(win):
    raise NotImplementedError

def vg_betweenees_centrality(win):
    raise NotImplementedError

def vg_scale_free(win):
    raise NotImplementedError

def vg_average_clustering_coefficient(win):
    raise NotImplementedError

def vg_diameter(win):
    raise NotImplementedError

def vg_global_efficiency(win):
    raise NotImplementedError

def vg_local_efficiency(win):
    raise NotImplementedError

def vg_mean_degree(win):
    raise NotImplementedError

def vg_transitivity(win):
    raise NotImplementedError
