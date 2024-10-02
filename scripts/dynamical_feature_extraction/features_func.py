import numpy as np
import itertools
from scipy.special import entr

def batch(iterable, size):
    it = iter(iterable)
    while item := list(itertools.islice(it, size)):
        yield item
        

def create_indexes(dimension_1, dimension_2, flatten = True):
    # Generate a meshgrid of indices and reshape to get pairs
    x, y = np.meshgrid(np.arange(dimension_1), np.arange(dimension_2), indexing='ij')
    indexes = np.array([x.flatten(), y.flatten()]).T
    if flatten:
        return indexes
    return batch(indexes, dimension_2)


def flatten_remove_diag(x):
    x_no_diag = np.ndarray.flatten(x)
    x_no_diag = np.delete(x_no_diag, range(0, len(x_no_diag), len(x) + 1), 0)
    return x_no_diag


def electrode_correlation(win):
    """
    Computes pairwise correlation between channels (electrodes) in a window of signal data.

    Parameters:
    win : numpy.ndarray
        2D array (time_steps, channels), where each column represents an electrode's signal over time.

    Returns:
    numpy.ndarray
        1D array containing upper triangle of the correlation matrix (excluding the diagonal).
        If the number of time steps is less than 2, returns an array of zeros.
    """
    n_channels = win.shape[1]
    T = win.shape[0]
    return [[0]] * (n_channels * (n_channels - 1) // 2) if T < 2 else np.corrcoef(win.T)[np.triu_indices(n_channels, k = 1)]


def calculate_auto_correlation(signal, time_step_lags, normalize=True):
    """
    Computes auto-correlation for a multi-channel signal with a specified time lag.

    Parameters:
    signal : numpy.ndarray
        2D array (time_steps, channels), where rows represent time points and columns represent channels.
    time_step_lags : int
        Time lag for calculating auto-correlation.
    subtract_mean : bool, optional
        Subtract the mean of each channel before calculating auto-correlation. Default is False.
    normalize : bool, optional
        Normalize by variance (auto-correlation at lag 0). Default is False.

    Returns:
    numpy.ndarray
        1D array of auto-correlation values for each channel.
    """
    corr = 0
    lagged_signal = signal[:-time_step_lags, :]  # Signal without the last 'time_step_lags' time points
    current_signal = signal[time_step_lags:, :]  # Signal without the first 'time_step_lags' time points
    
    # Element-wise multiplication and then averaging over the time dimension
    corr = np.mean(np.multiply(lagged_signal, current_signal), axis=0)
    
    if normalize:
        variance = np.mean(np.multiply(signal, signal), axis=0)  # Auto-correlation at lag 0
        corr /= variance
    return corr


def mutual_information_2d(win, padding_left, time_step_lags):
    """
    Computes mutual information between current time steps and lagged time steps for each channel.

    Parameters:
    win : numpy.ndarray
        2D array (time_steps, channels), where columns represent different channels' signals.
    padding_left : numpy.ndarray
        Padding array used for alignment before calculating correlations (typically past values).
    time_step_lags : int
        Time lag between current and past time steps for calculating correlations.

    Returns:
    numpy.ndarray
        A matrix of mutual information values between current and lagged signals for each channel.
    """
    channels = win.shape[1]
    time_steps = win.shape[0]
    win = np.vstack([padding_left, win]).T
    correlation_matrix = np.corrcoef(
        win[:, padding_left.shape[0]:],  # Current time window
        win[:, padding_left.shape[0] - time_step_lags: padding_left.shape[0] - time_step_lags + time_steps]  # Lagged window
    )[:channels, channels:]    
    mutual_information_2d = -0.5 * np.log(1 - correlation_matrix ** 2) # Gaussian approximation for mutual information
    return mutual_information_2d

def mutual_information_flatten(win, mutual_information_2d):
    """
    Flattens the mutual information matrix by removing the diagonal elements.

    Parameters:
    win : numpy.ndarray
        2D array (time_steps, channels), representing signal data (not used in computation).
    mutual_information_2d : numpy.ndarray
        2D array of mutual information values between current and lagged signals for each channel.

    Returns:
    numpy.ndarray
        1D array of mutual information values with diagonal elements removed.
    """
    mutual_information_flatten = flatten_remove_diag(mutual_information_2d)
    return mutual_information_flatten

def distribution_entropy(win, mutual_information_flatten):
    """
    Calculates the entropy of a distribution based on mutual information values.

    Parameters:
    win : numpy.ndarray
        2D array (time_steps, channels), representing signal data (not used in computation).
    mutual_information_flatten : numpy.ndarray
        1D array of mutual information values (normalized to form a distribution).

    Returns:
    float
        The entropy of the distribution derived from the mutual information values.
    """
    probabilities = mutual_information_flatten / np.sum(mutual_information_flatten)
    probabilities = np.where(probabilities == 0, 1e-9 , probabilities)
    return -np.sum([p_ij * np.log(p_ij) for p_ij in probabilities])

def network_entropy(win, mutual_information_2d):
    """
    Calculates the normalized network entropy based on mutual information values.

    Parameters:
    win : numpy.ndarray
        2D array (time_steps, channels), representing signal data (not used in computation).
    mutual_information_2d : numpy.ndarray
        2D array of mutual information values between current and lagged signals for each channel.

    Returns:
    numpy.ndarray
        Normalized network entropy values for each channel based on mutual information.
    """
    mutual_information_2d = mutual_information_2d.copy()
    np.fill_diagonal(mutual_information_2d, 0)
    s = np.repeat(np.sum(mutual_information_2d, axis = 0), win.shape[1] - 1)
    return flatten_remove_diag(mutual_information_2d) / s


def visibility_graph(win, channel_id):
    cnt_time_steps = len(win)
    
    graph = np.zeros((cnt_time_steps, cnt_time_steps))
    
    diff_y = win[:, np.newaxis] - win  
    diff_x = np.arange(cnt_time_steps)[:, np.newaxis] - np.arange(cnt_time_steps)

    slopes = np.divide(diff_y, diff_x, out=np.zeros_like(diff_y), where=diff_x != 0)  

    visibility_mask = np.ones((cnt_time_steps, cnt_time_steps), dtype=bool)

    # Create a mask for the pairs (i, j) and all k in (i, j)
    for i in range(cnt_time_steps):
        for j in range(i + 1, cnt_time_steps):
            # Slope from i to j
            slope_ij = slopes[i, j]
            
            # Check the slopes between i and j
            # We need to compare slopes[i, k] < slopes[i, j] for k in (i+1, j-1)
            # We create a mask for k in range(i+1 to j)
            visibility_mask[i, i + 1:j] &= (slopes[i, i+1:j] < slope_ij)
            # If any k fails the visibility, we mark (i, j) as not visible
            if not visibility_mask[i, j]:
                visibility_mask[i, j] = False
                visibility_mask[j, i] = False

    # Calculate the angles for visible edges
    graph = np.arctan(slopes)  # Calculate angles for all pairs
    graph[~visibility_mask] = 0  # Set angles to zero where points are not visible

    return graph

# CTE excludes indirect influences and is usually used to detect the direct causality
def conditional_transfer_information(win, padding_left, time_step_lags):
    """
    Calculates the Conditional Transfer Information (CTI) for direct causality detection
    among channels based on the input window data.

    Parameters:
    win : numpy.ndarray
        2D array (time_steps, channels) representing signal data.
    padding_left : numpy.ndarray
        1D array used for padding on the left side of the input data.
    time_step_lags : int
        Number of time steps to lag when calculating conditional information.

    Returns:
    numpy.ndarray
        A matrix representing the partial correlation coefficients between channels.
    """
    window_size = win.shape[0]
    n_channels = win.shape[1]

    win = np.vstack([padding_left, win])
    X_values = win[-time_step_lags - window_size:-time_step_lags, :][:, :, np.newaxis]
    Y_values = win[-window_size:, :][:, :, np.newaxis]
    bool_matrix = ~np.eye(n_channels, dtype=bool)
    Z_values = np.repeat(X_values, n_channels, axis = 2)[:, bool_matrix].reshape((-1, n_channels - 1, n_channels))
    Z_values = np.transpose(Z_values, axes=[0, 2, 1])

    Z = np.concatenate([X_values, Y_values, Z_values], axis=2)
    Z = Z[:, :, np.newaxis, :]  # Shape (T, C, 1, C + 1)
    Z = np.repeat(Z, n_channels, axis=2)  # Repeat along the electrode axis
    cov_pairwise = np.empty((n_channels, n_channels, n_channels + 1, n_channels + 1))

    for i in range(n_channels):
        for j in range(n_channels):
            cov_pairwise[i, j] = np.cov(Z[:, i, j, :], rowvar=False)
    return cov_pairwise[:, :, 0, 1] / np.sqrt(cov_pairwise[:, :, 0, 0] * cov_pairwise[:, :, 1, 1])


def adjacency_matrix(graph_matrix):
    return (np.abs(graph_matrix) > 0).astype(int)


def vg_degrees(win, visibility_graph, channel_id):
    """
    Calculates the degrees of all channels in the visibility graph.

    Parameters:
    visibility_graph : numpy.ndarray
        A 2D array representing the visibility graph where rows and columns correspond to channels.

    Returns:
    numpy.ndarray
        An array containing the degree of each channel, which is the number of direct connections it has.
    """
    vg = visibility_graph[channel_id]
    connection_matrix = adjacency_matrix(vg)
    degrees = np.sum(connection_matrix, axis = 0)
    return degrees
    

def vg_degree_local_entropy(win, vg_degrees, visibility_graph, channel_id):
    """
    Calculates the local entropy for each node in the visibility graph based on its neighbors' degrees.

    Parameters:
    win : numpy.ndarray
        2D array (time_steps, channels), representing signal data.
    vg_degrees : numpy.ndarray
        Array containing the degrees of each vertex in the visibility graph.
    visibility_graph : numpy.ndarray
        Matrix representing the visibility graph for the specified channel.
    channel_id : int
        The index of the channel to analyze.

    Returns:
    list
        A list of local entropy values for each node in the visibility graph.
    """
    def local_entropy_node_i(i):
        # Extract neighbors' degrees for the current node
        neighbor_out = visibility_graph[channel_id][i, :].reshape((-1))
        neighbor_in = visibility_graph[channel_id][:, i].reshape((-1))
        
        # Get degrees of neighboring nodes
        neighbor_degrees = np.array(vg_degrees[channel_id])[(np.abs(neighbor_out) > 0) | (np.abs(neighbor_in) > 0)]
        return entr(neighbor_degrees)
    return [local_entropy_node_i(i) for i in range(len(vg_degrees))]    


def vg_degree_distribution(win, vg_degrees, channel_id):
    """
    Computes the degree distribution for a specified channel in the visibility graph.

    Parameters:
    win : numpy.ndarray
        2D array (time_steps, channels), representing signal data.
    vg_degrees : numpy.ndarray
        Array containing the degrees of each vertex in the visibility graph for all channels.
    channel_id : int
        The index of the channel to analyze.

    Returns:
    numpy.ndarray
        The degree distribution for the specified channel, normalized by the number of time steps.
    """
    total_degrees = np.sum(vg_degrees[channel_id])
    if total_degrees == 0:
        return np.zeros_like(vg_degrees[channel_id]) 

    return vg_degrees[channel_id] / total_degrees


def vg_degree_entropy(win, vg_degree_distribution, channel_id):
    """
    Computes the entropy of the degree distribution for a specified channel in the visibility graph.

    Parameters:
    win : numpy.ndarray
        2D array (time_steps, channels), representing signal data.
    vg_degree_distribution : numpy.ndarray
        The degree distribution for all channels, where each entry is normalized.
    channel_id : int
        The index of the channel to analyze.

    Returns:
    float
        The entropy of the degree distribution for the specified channel.
    """
    def _vg_degree_entropy():
        distribution = np.where(vg_degree_distribution[channel_id] <= 0, 1e-9, vg_degree_distribution[channel_id])
        entropy_value = -np.sum(distribution * np.log(distribution))
        return entropy_value
    return _vg_degree_entropy()


def vg_degree_centrality(win, visibility_graph, channel_id):
    """
    Calculates the degree centrality of nodes in a visibility graph for a specified channel.

    Parameters:
    win : numpy.ndarray
        2D array (time_steps, channels), representing signal data.
    visibility_graph : numpy.ndarray
        A matrix representing the visibility graph from the time series data.
    channel_id : int
        The index of the channel for which to calculate degree centrality.

    Returns:
    numpy.ndarray
        An array containing the degree centrality for each node in the visibility graph.
    """
    def _vg_degree_centrality():
        direct_links = np.sum(visibility_graph[channel_id], axis = 1) #  Sum direct links for each node
        n = visibility_graph[channel_id].shape[0]  # Number of nodes
        return direct_links / ((n - 1) * (n - 2)) 
    return _vg_degree_centrality()


def vg_graph_index_complexity(win, visibility_graph, channel_id):
    """
    Calculates the graph index complexity of a visibility graph for a specified channel.

    Parameters:
    win : numpy.ndarray
        2D array (time_steps, channels), representing signal data.
    visibility_graph : numpy.ndarray
        A matrix representing the visibility graph from the time series data.
    channel_id : int
        The index of the channel for which to calculate graph index complexity.

    Returns:
    float
        The graph index complexity of the visibility graph for the specified channel.
    """
    def _vg_graph_index_complexity():
        n = visibility_graph[channel_id].shape[0] # Number of nodes in the visibility graph
        eigenvalues, _ = np.linalg.eig(visibility_graph[channel_id]) # Compute eigenvalues
        lambda_max = np.max(eigenvalues) # Maximum eigenvalue
        c = (lambda_max - 2 * np.cos(np.pi / (n - 1))) / (n - 1 - 2 * np.cos(np.pi / (n + 1)))
        return 4 * c * (1 - c)
    return _vg_graph_index_complexity()


def vg_u_graph_index_complexity(win, gic_window_size, recorder):
    gic_records = recorder['vg_graph_index_complexity'][:-gic_window_size]
    return np.mean(gic_records) + 3.1 * np.std(gic_records)


def vg_jaccard_similarity_coefficient(win, visibility_graph, channel_id):
    """
    Computes the Jaccard similarity coefficient matrix for the visibility graph of a given channel.

    Parameters:
    win : numpy.ndarray
        2D array (time_steps, channels), representing signal data.
    visibility_graph : numpy.ndarray
        The visibility graph for each channel.
    channel_id : int
        The index of the channel to analyze.

    Returns:
    numpy.ndarray
        A matrix representing the Jaccard similarity coefficients for each pair of nodes.
    """
    def _vg_jaccard_similarity_coefficient():
        # Number of nodes in the visibility graph for the given channel
        n = visibility_graph[channel_id].shape[0]
        vg = visibility_graph[channel_id]
        
        # Precompute neighbor selection matrix for all nodes
        neighbor_selection_matrix = (np.abs(vg) > 0) | (np.abs(vg.T) > 0)

        # Compute the intersection and union of neighbor sets for each pair of nodes
        intersection = neighbor_selection_matrix[:, :, np.newaxis] & neighbor_selection_matrix[:, np.newaxis, :]
        intersection_count = np.sum(intersection, axis=0)

        sum_neighbors = np.sum(neighbor_selection_matrix, axis=1)
        union_count = sum_neighbors[:, np.newaxis] + sum_neighbors[np.newaxis, :] - intersection_count

        # Calculate the Jaccard similarity coefficient matrix
        jaccard_similarity_matrix = intersection_count / np.where(union_count == 0, 1, union_count)

        return jaccard_similarity_matrix

    return _vg_jaccard_similarity_coefficient()
    
# # Pending:: need clustering
# def vg_modularity(win):
#     raise NotImplementedError

def vg_density(win, visibility_graph, channel_id):
    """
    Computes the density of the visibility graph for a given channel.

    Parameters:
    win : numpy.ndarray
        2D array (time_steps, channels), representing signal data.
    visibility_graph : numpy.ndarray
        The visibility graph for each channel.
    channel_id : int
        The index of the channel to analyze.

    Returns:
    float
        The density of the visibility graph.
    """
    def _vg_density():
        vg = visibility_graph[channel_id]
        cnt_edges = np.sum(np.abs(vg) > 0) // 2
        n = vg.shape[0]
        max_edges = n * (n - 1) / 2
        return cnt_edges / max_edges
    return _vg_density()
    

def vg_average_shortest_path_length(win, visibility_graph, full_source_full_destination_shortest_path, channel_id):
    """
    Computes the average shortest path length for the visibility graph of a given channel.

    Parameters:
    win : numpy.ndarray
        2D array (time_steps, channels), representing signal data.
    visibility_graph : numpy.ndarray
        The visibility graph for each channel.
    full_source_full_destination_shortest_path : numpy.ndarray
        Precomputed shortest path lengths for all pairs of nodes.
    channel_id : int
        The index of the channel to analyze.

    Returns:
    float
        The average shortest path length of the visibility graph.
    """
    def _vg_average_shortest_path_length():
        vg = visibility_graph[channel_id]
        distances = full_source_full_destination_shortest_path[channel_id]
        valid_distances = distances[np.isfinite(distances)]
        count_valid_pairs = len(valid_distances)
        if count_valid_pairs > 0:
            return np.sum(valid_distances) / count_valid_pairs
        else:
            return 0
    return _vg_average_shortest_path_length()


def vg_closeness_centrality(win, full_source_full_destination_shortest_path, channel_id):
    """
    Computes the closeness centrality for nodes in the visibility graph of a given channel.

    Parameters:
    win : numpy.ndarray
        2D array (time_steps, channels), representing signal data.
    full_source_full_destination_shortest_path : numpy.ndarray
        Precomputed shortest path lengths for all pairs of nodes.
    channel_id : int
        The index of the channel to analyze.

    Returns:
    numpy.ndarray
        An array of closeness centrality values for each node in the visibility graph.
    """
    def _vg_closeness_centrality():
        distances = full_source_full_destination_shortest_path[channel_id]
        distances[np.isinf(distances)] = 0
        n = distances.shape[0]
        
        result = (n - 1) / (np.sum(distances, axis = 1))
        result[np.isnan(result)] = 0
        return result
    return _vg_closeness_centrality()


def vg_betweenness_centrality(win, visibility_graph, channel_id):
    """
    Computes the betweenness centrality for each node in the visibility graph of the specified channel.

    Parameters:
    win : numpy.ndarray
        2D array (time_steps, channels), representing signal data.
    visibility_graph : numpy.ndarray
        3D array representing visibility graphs for all channels.
    channel_id : int
        The index of the channel to analyze.

    Returns:
    numpy.ndarray
        Array containing the betweenness centrality for each node in the visibility graph.
    """
    def _vg_betweenness_centrality():
        vg = visibility_graph[channel_id]
        n = vg.shape[0]
        betweenness = np.zeros(n)

        # Calculate shortest paths using the Floyd-Warshall algorithm
        distances = np.where(np.abs(vg) > 0, 1, np.inf)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    distances[i, j] = min(distances[i, j], distances[i, k] + distances[k, j])

        # Calculate betweenness centrality
        for s in range(n):
            for t in range(n):
                if s != t:
                    for u in range(n):
                        if u != s and u != t and distances[s, t] == distances[s, u] + distances[u, t]:
                            betweenness[u] += 1

        # Normalize by the number of pairs
        return betweenness / ((n - 1) * (n - 2))

    return _vg_betweenness_centrality()


def vg_diameter(win, full_source_full_destination_shortest_path, channel_id):
    """
    Computes the diameter of the visibility graph for the specified channel.

    Parameters:
    win : numpy.ndarray
        2D array (time_steps, channels), representing signal data.
    full_source_full_destination_shortest_path : numpy.ndarray
        3D array containing shortest path lengths for all source-destination pairs.
    channel_id : int
        The index of the channel to analyze.

    Returns:
    float
        The diameter of the visibility graph for the given channel.
    """
    return np.max(full_source_full_destination_shortest_path[channel_id])


def vg_global_efficiency(win, visibility_graph, full_source_full_destination_shortest_path, channel_id):
    """
    Computes the global efficiency of the visibility graph for a specified channel.

    Parameters:
    win : numpy.ndarray
        2D array (time_steps, channels), representing signal data.
    visibility_graph : numpy.ndarray
        3D array representing visibility graphs for all channels.
    full_source_full_destination_shortest_path : numpy.ndarray
        3D array of shortest paths for each channel.
    channel_id : int
        The index of the channel to analyze.

    Returns:
    float
        The global efficiency of the visibility graph for the specified channel.
    """
    Q = full_source_full_destination_shortest_path[channel_id].shape[0]
    distances = full_source_full_destination_shortest_path[channel_id]
    # Calculate the inverse of distances
    L = 1 / distances
    
    # Set infinities and NaNs to 0
    L[np.isinf(L)] = 0
    L[np.isnan(L)] = 0

    factor = 0 if Q <= 1 else (1 / (Q * (Q - 1)))
    
    return  factor * np.sum(L)


def vg_local_efficiency(win, full_source_full_destination_shortest_path, channel_id):
    """
    Calculates the local efficiency of the visibility graph for a specified channel.

    Parameters:
    win : numpy.ndarray
        2D array representing the signal data (not used in this function).
    full_source_full_destination_shortest_path : numpy.ndarray(2D)
        full_source_full_destination_shortest_path[i, j] store the distance from node i to node j.
    channel_id : int
        The index of the channel for which to calculate the mean degree.

    Returns:
    NDArray[1D]:
        The local efficiency of each node in visibility graph.
    """
    def _vg_local_efficiency():
        distances = full_source_full_destination_shortest_path[channel_id]
        local_efficiencies = []

        # Iterate through each node to calculate its local efficiency
        for i in range(distances.shape[0]):
            # Get neighbors: nodes not equal to the current node
            neighbors = np.arange(distances.shape[0]) != i
            neighbor_distances = distances[neighbors][:, neighbors]

            # Calculate local efficiency for node i
            if neighbor_distances.size > 0:
                L = 1 / neighbor_distances
                L[np.isinf(L)] = 0  # Handle infinite distances
                local_efficiency = np.sum(L) / (len(neighbors) * (len(neighbors) - 1)) if len(neighbors) > 1 else 0
            else:
                local_efficiency = 0
            
            local_efficiencies.append(local_efficiency)

        return np.array(local_efficiencies)

    return _vg_local_efficiency()


def vg_mean_degree(win, vg_degrees, channel_id):
    """
    Calculates the mean degree of the visibility graph for a specified channel.

    Parameters:
    win : numpy.ndarray
        2D array representing the signal data (not used in this function).
    vg_degrees : numpy.ndarray
        Array containing the degrees of nodes in the visibility graph for different channels.
    channel_id : int
        The index of the channel for which to calculate the mean degree.

    Returns:
    float
        The mean degree of the visibility graph for the specified channel.
    """
    def _vg_mean_degree():
        return np.average(vg_degrees[channel_id])
    return _vg_mean_degree()


def vg_average_weighted_degree(win, visibility_graph, channel_id):
    """
    Calculates the average weighted degree of the visibility graph for a specified channel.

    Parameters:
    win : numpy.ndarray
        2D array representing the signal data (not used in this function).
    visibility_graph : numpy.ndarray
        A matrix representing the weights (angles) for edges between time steps in the visibility graph.
    channel_id : int
        The index of the channel for which to calculate the average weighted degree.

    Returns:
    float
        The average weighted degree of the visibility graph for the specified channel.
    """
    # Sum the weighted degrees for the specified channel
    def _vg_average_weighted_degree():
        vg = visibility_graph[channel_id]
        return np.sum(vg, axis = 0) 
    return _vg_average_weighted_degree()


def vg_transitivity(win, visibility_graph, channel_id):
    def _vg_transitivity():
        vg = visibility_graph[channel_id]
        n = vg.shape[0]
        
        # Square the adjacency matrix to find 2-step connections
        A2 = np.dot(vg, vg)
        
        # Count Triangle
        A3 = np.dot(A2, vg)
        
        N_delta = np.trace(A3) / 6  # Each triangle is counted 6 times in A3
        # N_V counts the number of pairs of connected nodes (edges)
        N_V = np.sum(vg) / 2  # Each edge is counted twice
        
        return (3 * N_delta / N_V) if N_V > 0 else 0  # Prevent division by zero

    return _vg_transitivity()

def vg_similarity(win):
    raise NotImplementedError


def floyd_warshall(graph_matrix):
    """
    Applies the Floyd-Warshall algorithm to find the shortest paths between all pairs of nodes.

    Parameters:
    graph_matrix : numpy.ndarray
        A square 2D array representing the weighted adjacency matrix of the graph.
        Non-connections are assumed to be np.inf.

    Returns:
    numpy.ndarray
        A square 2D array containing the shortest path distances between all pairs of nodes.
    """
    n = graph_matrix.shape[0]
    distance = np.copy(graph_matrix)  # Start with the adjacency matrix
    distance[distance == 0] = np.inf
    # Set diagonal to zero (distance to self)
    np.fill_diagonal(distance, 0)
    
    # Matrix-based update of distances
    for k in range(n):
        distance = np.minimum(distance, distance[:, k][:, np.newaxis] + distance[np.newaxis, k, :])
    
    return distance


def full_source_full_destination_shortest_path(win, visibility_graph, channel_id):
    """
    Find the shortest paths between all pairs of nodes.

    Parameters:
    visibility_graph : numpy.ndarray
        A square 2D array representing the visibility_graph.
    channel_id: int
        An int channel id that denote which specific EEG channel's visibility graph should be processed.

    Returns:
    numpy.ndarray
        A square 2D array containing the shortest path distances between all pairs of nodes.
    """
    return floyd_warshall(visibility_graph[channel_id])
    

def reconstructed_phase_space(win, delay_time_steps, padding_left=None):
    """
    Reconstruct the phase space of a multi-channel time series data.

    Parameters:
    win : numpy.ndarray
        A 2D array where each row represents a time step and each column represents a channel.
    delay_time_steps : int
        The number of previous time steps to consider for each state vector.
    padding_left : numpy.ndarray, optional
        An array of values used to fill in the state vectors when there are not enough previous time steps.

    Returns:
    numpy.ndarray
        A 2D array representing the reconstructed phase space, where each row corresponds to a state vector.
    """
    T = win.shape[0]
    n_channels = win.shape[1]
    state_vectors = np.zeros((T, delay_time_steps + 1, n_channels))
    for t in range(T):
            if t < delay_time_steps:
                if padding_left is not None:
                    state_vectors[t, :delay_time_steps - t] = padding_left[-(delay_time_steps - t):]
                state_vectors[t, delay_time_steps - t:] = win[:t + 1]
            else:
                state_vectors[t, :] = win[t - delay_time_steps: t + 1]
    return state_vectors.reshape((T, -1))


def recurrence_plot(win, reconstructed_phase_space, distance_threshold):
    """
    Generates a recurrence plot based on the reconstructed phase space of time series data.

    Parameters:
    win : numpy.ndarray
        A 2D array where each row represents a time step and each column represents a channel.
        (Note: This parameter is not used directly in the function.)
    reconstructed_phase_space : numpy.ndarray
        A 2D array where each row corresponds to a state vector from the reconstructed phase space.
    distance_threshold : float
        The distance threshold for defining recurrences; pairs of points with distances less than or equal to this
        threshold will be marked as recurrent.

    Returns:
    numpy.ndarray
        A binary recurrence plot where elements are 1 if the distance between corresponding points is less than
        or equal to the distance threshold, and 0 otherwise.
    """
    T = win.shape[0]  # Number of time steps
    # Compute the pairwise distance matrix for the phase space vectors
    distances = np.linalg.norm(reconstructed_phase_space[:, np.newaxis, :] - reconstructed_phase_space[np.newaxis, :, :], axis=2)

    # Generate the recurrence plot by applying the distance threshold
    recurrence_plot = (distances <= distance_threshold).astype(float) 

    return recurrence_plot


def rp_recurrence_rate(win, recurrence_plot):
    """
    Calculates the recurrence rate from a given recurrence plot.

    Parameters:
    win : numpy.ndarray
        A 2D array representing time series data (not used in this calculation).
    recurrence_plot : numpy.ndarray
        A 2D binary array where each element indicates whether a state is recurrent (1) or not (0).

    Returns:
    float
        The recurrence rate, representing the proportion of recurrent states in the recurrence plot.
    """
    if recurrence_plot.ndim != 2 or recurrence_plot.shape[0] != recurrence_plot.shape[1]:
        raise ValueError("recurrence_plot must be a square 2D binary array.")
    
    N = recurrence_plot.shape[0]
    return (1 / (N * N)) * np.sum(recurrence_plot)


def diagnoal_length_distribution(win, recurrence_plot):
    """
    Calculates the distribution of diagonal lengths in a recurrence plot.

    Parameters:
    win : numpy.ndarray
        A 2D array representing time series data (not used in this calculation).
    recurrence_plot : numpy.ndarray
        A 2D binary array where each element indicates whether a state is recurrent (1) or not (0).

    Returns:
    numpy.ndarray
        An array representing the normalized distribution of diagonal lengths.
    """
    #  Validate input
    if recurrence_plot.ndim != 2 or recurrence_plot.shape[0] != recurrence_plot.shape[1]:
        raise ValueError("recurrence_plot must be a square 2D binary array.")
    N = recurrence_plot.shape[0]
    length_distribution = np.zeros((N + 1))
    for i in range(-N + 1, N):
        # Iterate i-th diagonal.
        current_contiguous_points = 0
        for d in range(N - np.abs(i)):
            x_coordination = d if N >= 0 else -i + d
            y_coordination = i + d if N >= 0 else d
            if recurrence_plot[x_coordination, y_coordination] == 1:
                current_contiguous_points += 1
            else:
                length_distribution[current_contiguous_points] += 1
                current_contiguous_points = 0
            
        if current_contiguous_points > 0:
            length_distribution[current_contiguous_points] += 1
            
            
    length_distribution_sum = sum(length_distribution)
    return length_distribution / (length_distribution_sum if length_distribution_sum != 0 else 1e-9)


def vertical_length_distribution(win, recurrence_plot):
    """
    Calculates the distribution of vertical lengths in a recurrence plot.

    Parameters:
    win : numpy.ndarray
        A 2D array representing time series data (not used in this calculation).
    recurrence_plot : numpy.ndarray
        A 2D binary array where each element indicates whether a state is recurrent (1) or not (0).

    Returns:
    numpy.ndarray
        An array representing the normalized distribution of vertical lengths.
    """
    N = recurrence_plot.shape[0]
    length_distribution = np.zeros((N + 1))
    for i in range(N):
        current_contiguous_points = 0
        for j in range(N):
            if recurrence_plot[j, i] == 1:
                current_contiguous_points += 1
            else:
                length_distribution[current_contiguous_points] += 1
                current_contiguous_points = 0
        if current_contiguous_points != 0:
            length_distribution[current_contiguous_points] += 1
            
    length_distribution_sum =  sum(length_distribution)
    return length_distribution / (length_distribution_sum if length_distribution_sum != 0 else 1e-9)


def rp_determinism(win, diagnoal_length_distribution, l_min):
    """
    Calculates the determinism of a recurrence plot.

    Parameters:
    win : numpy.ndarray
        A 2D array representing time series data (not used in this calculation).
    diagnoal_length_distribution : numpy.ndarray
        An array representing the normalized distribution of diagonal lengths in the recurrence plot.
    l_min : int
        The minimum length of diagonal lines to be considered for the determinism calculation.

    Returns:
    float
        The determinism value, which is the ratio of the weighted lengths of diagonal lines 
        greater than or equal to l_min to the total weighted length of all diagonal lines.
    """
    L =  diagnoal_length_distribution.shape[0]
    weighted_distribution = np.multiply(np.arange(L), diagnoal_length_distribution)
    weighted_distribution_sum_l_min_1 = np.sum(weighted_distribution[1:])
    return 0 if weighted_distribution_sum_l_min_1 == 0 else np.sum(weighted_distribution[l_min:]) / weighted_distribution_sum_l_min_1


def rp_average_diagonal_line_length(win, diagnoal_length_distribution, l_min):
    """
    Calculates the average length of diagonal lines in a recurrence plot.

    Parameters:
    win : numpy.ndarray
        A 2D array representing time series data (not used in this calculation).
    diagnoal_length_distribution : numpy.ndarray
        An array representing the normalized distribution of diagonal lengths in the recurrence plot.
    l_min : int
        The minimum length of diagonal lines to be considered for the average calculation.

    Returns:
    float
        The average length of diagonal lines greater than or equal to l_min. Returns 0 if no valid lines are found.
    """
    L = diagnoal_length_distribution.shape[0]
    weighted_distribution = np.multiply(np.arange(L), diagnoal_length_distribution)
    diagnoal_length_distribution_sum = np.sum(diagnoal_length_distribution[l_min:])
    return 0 if diagnoal_length_distribution_sum == 0 else np.sum(weighted_distribution[l_min:]) / diagnoal_length_distribution_sum


def rp_longest_diagonal_line(win, diagnoal_length_distribution):
    """
    Finds the length of the longest diagonal line in a recurrence plot.

    Parameters:
    win : numpy.ndarray
        A 2D array representing time series data (not used in this calculation).
    diagnoal_length_distribution : numpy.ndarray
        An array representing the normalized distribution of diagonal lengths in the recurrence plot.

    Returns:
    int
        The length of the longest diagonal line. Returns -1 if no diagonal lines exist.
    """
    return np.argmax(np.arange(0, len(diagnoal_length_distribution)) * (diagnoal_length_distribution > 0))


def rp_entropy_of_diagonal_lines(win, diagnoal_length_distribution):
    """
    Calculates the entropy of diagonal lines in a recurrence plot based on the diagonal length distribution.

    Parameters:
    win : numpy.ndarray
        A 2D array representing time series data (not used in this calculation).
    diagnoal_length_distribution : numpy.ndarray
        An array representing the normalized distribution of diagonal lengths in the recurrence plot.

    Returns:
    float
        The entropy value representing the uncertainty or complexity of the diagonal lines in the recurrence plot.
    """
    entropy = -sum(p * np.log(p) for p in diagnoal_length_distribution if p > 0)
    return entropy


def rp_laminarity(win, vertical_length_distribution, l_min = 2):
    """
    Calculates the laminarity in a recurrence plot based on the vertical length distribution.

    Parameters:
    win : numpy.ndarray
        A 2D array representing time series data (not used in this calculation).
    vertical_length_distribution : numpy.ndarray
        An array representing the normalized distribution of vertical lengths in the recurrence plot.
    l_min : int, optional
        The minimum vertical length to consider for calculating the laminarity (default is 2).

    Returns:
    float
        The average laminarity calculated from the weighted distribution of vertical lengths.
        Returns 0 if the sum of the weighted distribution for lengths greater than or equal to l_min is zero.
    """
    L = vertical_length_distribution.shape[0]
    weighted_distribution = np.multiply(np.arange(L), vertical_length_distribution)
    weighted_distribution_sum = np.sum(weighted_distribution[l_min:])
    return 0 if weighted_distribution_sum == 0 else np.sum(weighted_distribution[l_min:]) / weighted_distribution_sum


def rp_trapping_time(win, diagnoal_length_distribution, l_min = 2):
    """
    Calculates the trapping time in a recurrence plot based on the diagonal length distribution.

    Parameters:
    win : numpy.ndarray
        A 2D array representing time series data (not used in this calculation).
    diagnoal_length_distribution : numpy.ndarray
        An array representing the normalized distribution of diagonal lengths.
    l_min : int
        The minimum diagonal length to consider for calculating the trapping time.

    Returns:
    float
        The average trapping time calculated from the weighted distribution of diagonal lengths.
        Returns 0 if the sum of the distribution for lengths greater than or equal to l_min is zero.
    """
    L = diagnoal_length_distribution.shape[0]
    weighted_diagnoal_length_distribution = np.multiply(np.arange(L), diagnoal_length_distribution)
    diagnoal_length_distribution_sum = np.sum(diagnoal_length_distribution[l_min:])
    return 0 if diagnoal_length_distribution_sum == 0 else np.sum(weighted_diagnoal_length_distribution[l_min:]) / diagnoal_length_distribution_sum

# Other features

def katz_fractal_dimensions(win):
    diff = (np.roll(win, -1) - win)[:-1]
    N = win.shape[0]
    L = np.sum(np.abs(diff), axis = 0)
    d = np.max(np.abs(win - win[0, :]), axis = 0)
    d = np.where(d <= 0, 1e-9, d)  
    L = np.where(L <= 0, 1e-9, L)  
    logN = np.log(N)
    ratio = np.log(d) - np.log(L)
    return logN / (logN + ratio)

def electrode_sd(win):
    return np.std(win, axis = 0)