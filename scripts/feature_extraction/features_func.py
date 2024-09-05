import numpy as np
import itertools

def batch(iterable, size):
    it = iter(iterable)
    while item := list(itertools.islice(it, size)):
        yield item
        
def create_indexes(dimension_1, dimension_2, flatten = True):
    def cartesian_product(x, y):
        import itertools
        return itertools.product(x, y)
    indexes = list(cartesian_product(range(dimension_1), range(dimension_2)))
    if flatten:
        return indexes
    return batch(indexes, dimension_2)

def flatten_remove_diag(x):
    x_no_diag = np.ndarray.flatten(x)
    x_no_diag = np.delete(x_no_diag, range(0, len(x_no_diag), len(x) + 1), 0)
    x_no_diag = x_no_diag.reshape(len(x), len(x) - 1)
    return x_no_diag.ravel()

def electrode_correlation(win):
    n_channels = win.shape[1]
    return [[0]] * (n_channels * (n_channels - 1) // 2) if win.shape[0] < 2 else np.corrcoef(win.T)[np.triu_indices(n_channels, k = 1)]

def calculate_auto_correlation(signal, time_step_lags):
    corr = 0
    for t in range(time_step_lags, signal.shape[0]):
        corr += np.multiply(signal[t - time_step_lags, :], signal[t, :])
    return corr / (signal.shape[0] - time_step_lags)

def mutual_information_2d(win, padding, time_step_lag):
    channels = win.shape[1]
    win_length = win.shape[0]
    win = np.vstack([padding, win]).T
    correlation_matrix = np.corrcoef(win[:, padding.shape[0]:], win[:, padding.shape[0]-time_step_lag: padding.shape[0]-time_step_lag + win_length])[:channels, channels:]
    mutual_information_2d = (-0.5 * (1 - (correlation_matrix) ** 2))
    return mutual_information_2d

def mutual_information_flatten(win, mutual_information_2d):
    mutual_information_flatten = flatten_remove_diag(mutual_information_2d)
    return mutual_information_flatten

def distribution_entropy(win, mutual_information):
    return np.sum([p_ij * np.log(p_ij) for p_ij in mutual_information / np.sum(mutual_information)])

def network_entropy(win, mutual_information_2d):
    mutual_information_2d = mutual_information_2d.copy()
    np.fill_diagonal(mutual_information_2d, 0)
    s = np.repeat(np.sum(mutual_information_2d, axis = 0), win.shape[1] - 1)
    return flatten_remove_diag(mutual_information_2d) / s

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    
# TODE: optimize the complexity (segment tree may optimize it.) 
def visibility_graph(win):
    cnt_time_steps = len(win)
    # value_diff = np.repeat(data[:,np.newaxis].T, cnt_time_steps, axis=0) - np.repeat(data[:,np.newaxis], cnt_time_steps, axis=1)
    # time_differ = np.repeat(np.arange(0, cnt_time_steps)[:,np.newaxis].T, cnt_time_steps, axis=0) - np.repeat(np.arange(0, cnt_time_steps)[:,np.newaxis], cnt_time_steps, axis=1)
    # graph = np.arctan(value_diff / time_differ)
    indexes = create_indexes(cnt_time_steps, cnt_time_steps, flatten=True)
    graph = ([0 if index[0] >= index[1] or any(x >= (win[index[0]] + ((index[0] + 1 + t_c - index[0]) / (index[1] - index[0])) * \
            (win[index[1]] - win[index[0]])) for t_c, x in enumerate(win[index[0] + 1: index[1]])) else \
            np.arctan((win[index[1]] - win[index[0]]) / (index[1] - index[0])) for index in indexes])
    graph = np.array(graph).reshape(cnt_time_steps, cnt_time_steps)
    return graph

# CTE excludes indirect influences and is usually used to detect the direct causality
def conditional_transfer_information(win, left_padding, time_step_lags):
    window_size = win.shape[0]
    cnt_time_steps = window_size
    cnt_channels = win.shape[1]
    indexes = create_indexes(cnt_channels, cnt_channels)
    win = np.vstack([left_padding, win]).T
    padding_length = left_padding.shape[0]
    def calculate_partial_correlation_rho():
        def _calculate_partial_correlation_rho(x_id, y_id):
            Z = np.vstack([
                np.array(win[:, -(time_step_lags + window_size):-(time_step_lags)][x_id]), \
                np.array(win[:, padding_length:][y_id]), \
                np.delete(win[:, -(time_step_lags + window_size):-(time_step_lags)], (0), axis = 0)])
            vars = np.cov(Z)
            return vars[0, 1] / np.sqrt((1 / vars[0, 0]) * (1 / vars[1, 1]))
        rho = np.array([_calculate_partial_correlation_rho(index[0], index[1]) for index in indexes])
        return rho
    return np.array(calculate_partial_correlation_rho()).reshape(win.shape[0], win.shape[0])

def vg_degree(win, visibility_graph):
    vg = visibility_graph
    connection_matrix = [(vg > 0.1)]
    return np.sum(connection_matrix +
                    np.zeros((connection_matrix.shape[0], connection_matrix.shape[1])),
                    axis = 0)
    
def vg_degree_local_entropy(win, vg_degree, visibility_graph):
    def local_entropy_node_i(i):
        from scipy.special import entr
        neighbor_out = visibility_graph[i, :]
        neighbor_in = visibility_graph[:, i]
        node_select = (neighbor_out > 0) or (neighbor_in > 0)
        neighbor_degrees =  vg_degree[node_select]
        return entr(neighbor_degrees)
    cnt_nodes = vg_degree.shape[0]
    return np.array([local_entropy_node_i(i) for i in range(cnt_nodes)])
    
def vg_degree_distribution(win, vg_degree):
    return vg_degree / vg_degree.shape[0]
    
def vg_degree_entropy(win, vg_degree_distribution):
    def _vg_degree_entropy():
        return np.sum(np.multiply(vg_degree_distribution, np.log(vg_degree_distribution)))
    return _vg_degree_entropy()

def vg_degree_centrality(win, vg_degree):
    def _vg_degree_centrality():
        direct_links = np.sum(vg_degree, axis = 1)
        n = vg_degree.shape[0]
        return direct_links / ((n - 1) * (n - 2)) 
    return _vg_degree_centrality()
            
def vg_graph_index_complexities(win, visibility_graph):
    def _vg_graph_index_complexity():
        n = visibility_graph.shape[0]
        eigenvalues, _ = np.linalg.eig(visibility_graph)
        lambda_max = np.max(eigenvalues)
        c = (lambda_max - 2 * np.cos(np.pi / (n - 1))) / (n - 1 - 2 * np.cos(np.pi / (n + 1)))
        return 4 * c * (1 - c)
    return _vg_graph_index_complexity()

def vg_jaccard_similarity_coefficient(win, visibility_graph):
    def _vg_jaccard_similarity_coefficient():
        n = visibility_graph.shape[0]
        vg = visibility_graph
        def _neighbor_selection_matrix(node_id):
            return (vg[node_id, :] > 0) or (vg[:, node_id] > 0)
        S = np.array([ np.sum(_neighbor_selection_matrix(index[0]) and _neighbor_selection_matrix(index[1])) \
            /  (np.sum(_neighbor_selection_matrix(index[0])) + np.sum(_neighbor_selection_matrix(index[1])) -  np.sum(_neighbor_selection_matrix(index[0]) and _neighbor_selection_matrix(index[1]))) \
            for index in create_indexes(n, n)]).reshape(n, n)
        return S
    return _vg_jaccard_similarity_coefficient()   
    
# # Pending:: need clustering
# def vg_modularity(win):
#     raise NotImplementedError

def vg_density(win, visibility_graph):
    def _vg_density():
        vg = visibility_graph
        cnt_edges = np.sum((vg > 0.1) + np.zeros((vg.shape[0], vg.shape[1]))) // 2
        n = vg.shape[0]
        return 2 * cnt_edges / n * (n * 1)
    return _vg_density()
    
def vg_average_shortest_path_length(win, visibility_graph, full_source_full_destination_shortest_path):
    def _vg_average_shortest_path_length():
        vg = visibility_graph
        distances, _ = full_source_full_destination_shortest_path
        n = vg.shape[0]
        return (distances) / (n * (n - 1))
    return _vg_average_shortest_path_length()

def vg_closeness_centrality(win, full_source_full_destination_shortest_path):
    def _vg_closeness_centrality():
        distances, parents = full_source_full_destination_shortest_path
        n = full_source_full_destination_shortest_path.shape[0]
        return (n - 1) / (np.sum(distances, axis = 1))
    return _vg_closeness_centrality()

# # TODO: n = vertexes..
# def vg_small_worlds(win, visibility_graphs, full_source_full_destination_shortest_paths):
#     def _vg_small_world(graph_id):
#         L = np.average(np.nan_to_num(full_source_full_destination_shortest_paths[graph_id], posinf=0))
#         n = visibility_graphs[graph_id].shape[0]
#         lnN = np.log(np.sum(visibility_graphs[visibility_graphs > 0] + np.zeros((n, n))))

# def vg_eigenvector_centrality(win):
#     pass

def vg_betweenees_centrality(win, visibility_graph):
    def _vg_betweenees_centrality():
        def get_betweeness(vg):
            betweeness = np.zeros((vg.shape[0], vg.shape[1]))
            for i in range(vg.shape[0]):
                for j in range(vg.shape[1]):
                    for k in range(vg.shape[0]):
                        if k == i or k == j:
                            continue
                        if vg[i, k] > 0 and vg[k, j] > 0:
                            betweeness[i, j] += 1
        vg = visibility_graph
        betweeness = get_betweeness(vg)
        n = vg.shape[0]
        return np.sum((2 * betweeness / ((n - 1) * (n - 1))), axis = 1)
    return _vg_betweenees_centrality()

def vg_diameter(win, full_source_full_destination_shortest_path):
    return np.max(full_source_full_destination_shortest_path)

def vg_global_efficiency(win, full_source_full_destination_shortest_path):
    Q = full_source_full_destination_shortest_path.shape[0]
    return (1 / (Q * (Q - 1))) * np.sum(1 / full_source_full_destination_shortest_path)

def vg_local_efficiency(win, full_source_full_destination_shortest_path):
    Q = full_source_full_destination_shortest_path.shape[0]
    def _vg_local_efficiency():
        return (1 / (Q * (Q - 1))) * \
            np.sum(1 / full_source_full_destination_shortest_path, axis=1) 
    return _vg_local_efficiency()

def vg_mean_degree(win, vg_degree):
    def _vg_mean_degree():
        return np.average(vg_degree)
    return _vg_mean_degree()
    
def vg_avergae_weighted_degree(win, visibility_graph):
    def _vg_avergae_weighted_degree():
        return np.sum(visibility_graph, axis = 0) + np.sum(visibility_graph, axis = 1)
    return _vg_avergae_weighted_degree()

def vg_transitivity(win, visibility_graph):
    def _vg_transitivity():
        max_cnt_nodes = visibility_graph.shape[0]
        N_delta = 0
        N_V = 0
        vg = visibility_graph
        for i in range(max_cnt_nodes):
            for j in range(max_cnt_nodes):
                for k in range(max_cnt_nodes):
                    c = 0
                    c += 1 if (vg[i, j] > 0) else 0
                    c += 1 if (vg[j, k] > 0) else 0
                    c += 1 if (vg[i, k] > 0) else 0
                    if c == 3:
                        N_delta += 1
                    elif c == 2:
                        N_V += 1
        return 3 * N_delta / N_V
    return _vg_transitivity()

def vg_similarity(win):
    raise NotImplementedError

def dijstra_full_source_full_destination(graph_matrix):
    n = graph_matrix.shape[0]
    distance = np.ones((n, n)) * np.inf
    parents = np.ones((n, n))
    for i in range(n):
        distance[i, i] = 0
        parents[i, i] = i
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if(graph_matrix[i, k] + graph_matrix[j, k] < distance[i, k]):
                    distance[i, k] = graph_matrix[i, k] + graph_matrix[j, k]
                    parents[i, j] = k
    return distance, parents

def full_source_full_destination_shortest_path(win, visibility_graph):
    return dijstra_full_source_full_destination(visibility_graph)
    
def reconstructed_phase_space(win, delay_time_steps, padding_left=None):
    T = win.shape[0]
    state_vectors = np.zeros((T, delay_time_steps + 1))
    for t in range(T):
            if t < delay_time_steps:
                if padding_left is not None:
                    state_vectors[t, :delay_time_steps - t] = padding_left[-(delay_time_steps - t):]
                state_vectors[t, delay_time_steps - t:] = win[:t + 1]
            else:
                state_vectors[t, :] = win[t - delay_time_steps: t + 1]
    return state_vectors    

## Recurrance plot-based features.
def recurrance_plot(data, reconstructed_phase_space, distance_threshold):
    row_indexes = np.repeat(list(range(data.shape[0])), axis=0)
    col_indexes = np.repeat(list(range(data.shape[0])), axis=1)
    recurrance_plot = map(lambda corrdination: np.sqrt(np.sum(reconstructed_phase_space[corrdination[0],:] - reconstructed_phase_space[corrdination[1],:]) ** 2) , zip(row_indexes, col_indexes))
    return recurrance_plot[recurrance_plot >= distance_threshold] + np.zeros((recurrance_plot.shape[0], recurrance_plot.shape[1]))    

def rp_recurrence_rate(win, recurrance_plot):
    N = recurrance_plot.shape[0]
    return (1 / (N * N)) * np.sum(recurrance_plot)

def diagnoal_length_distribution(win, recurrance_plot):
    N = recurrance_plot.shape[0]
    length_distribution = np.zeros((N + 1))
    for i in range(N):
        current_contiguous_points = 0
        for j in range(i):
            if recurrance_plot[i - j, j] == 1:
                current_contiguous_points += 1
            if j == 0:
                length_distribution[current_contiguous_points] += 1
                current_contiguous_points = 0
        if current_contiguous_points != 0:
            length_distribution[current_contiguous_points] += 1
    return length_distribution / sum(length_distribution)
    

def vertical_length_distribution(win, recurrance_plot):
    N = recurrance_plot.shape[0]
    length_distribution = np.zeros((N + 1))
    for i in range(0, N):
        current_contiguous_points = 0
        for j in range(N):
            if recurrance_plot[j, i] == 1:
                current_contiguous_points += 1
            if j == 0:
                length_distribution[current_contiguous_points] += 1
                current_contiguous_points = 0
        if current_contiguous_points != 0:
            length_distribution[current_contiguous_points] += 1
            
    return length_distribution / sum(length_distribution)

def rp_determinism(win, diagnoal_length_distribution, l_min):
    weighted_distribution = \
        np.arange(0, diagnoal_length_distribution.shape[0]).multiply(diagnoal_length_distribution)
    return np.sum(weighted_distribution[l_min:]) / np.sum(weighted_distribution)

def rp_average_diagonal_line_length(win, diagnoal_length_distribution, l_min):
    weighted_distribution = \
        np.arange(0, diagnoal_length_distribution.shape[0]).multiply(diagnoal_length_distribution)
    return np.sum(weighted_distribution[l_min:]) / np.sum(diagnoal_length_distribution[l_min:])

def rp_longest_diagonal_line(win, diagnoal_length_distribution):
    return np.argmax(np.arange(0, len(diagnoal_length_distribution)) * (diagnoal_length_distribution > 0))

def rp_entropy_of_diagonal_lines(win, diagnoal_length_distribution):
    from scipy.special import entr
    return entr(diagnoal_length_distribution)

def rp_laminarity(win, vertical_length_distribution, l_min):
    weighted_distribution = \
        np.arange(0, vertical_length_distribution.shape[0]).multiply(vertical_length_distribution)
    return np.sum(weighted_distribution[l_min:]) / np.sum(weighted_distribution)

def rp_trapping_time(win, length_distribution, l_min):
    weighted_distribution = \
        np.arange(0, length_distribution.shape[0]).multiply(length_distribution)
    return np.sum(weighted_distribution[l_min:]) / np.sum(length_distribution[l_min:])

# Other features
def katz_fractal_dimensions(win):
    diff = (np.roll(win, -1) - win)[:-1]
    N = win.shape[1]
    L = np.sum(diff, axis = 1)
    d = np.max(np.abs(win - win[0, :]), axis = 1)
    return np.log(N) / (np.log(N) + np.log(d / L))

def electrode_sd(win):
    return np.std(win, axis = 0)


