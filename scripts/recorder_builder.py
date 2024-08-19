import numpy as np
import os, sys
sys.path.append(".")

from feature_extractor import RecordConfiguration
from features_func import *


def build_recorder_config(n_channels, configuration={}):
    config = RecordConfiguration()
    
    # Dynamic network-based features
    config.add_record_object("electrode_sd", lambda win: np.std(win, axis = 0))
    config.add_record_object('electrode_local_entropy', lambda win: electrode_local_entropy(win, n_channels))

    config.add_record_object('electrode_corr', lambda win: electrode_correlation(win, n_channels))
    config.add_record_object("network_entropy", lambda win: network_entropy(win, time_step_lag=
        configuration.get("network_entropy_time_lags", 15)))
    config.add_record_object("conditional_transfer_information", lambda win: conditional_transfer_information(win, 
        time_step_lag=configuration.get("conditional_transfer_information_time_lags", 15)))
    config.add_record_object("mutual_information", lambda win: mutual_information(win, 
        time_step_lag=configuration.get("mutual_information_time_lags", 15)))
    config.add_record_object("distribution_entropy", lambda win: distribution_entropy(win))
    
    # Visibility graph-based features.
    config.add_record_object("visibility_graph", lambda win:
        [visuality_graph(win[i,:]) for i in range(win.shape[1])], temporary = True)
    
    config.add_record_object("graph_index_complexity", lambda win: vg_graph_index_complexity(win), 
                             dependencies=['visibility_graph'])
    config.add_record_object("degree_distribution", lambda win: vg_degree_distribution(win), 
                             dependencies=['visibility_graph'])
    config.add_record_object("degree_entropy", lambda win: vg_degree_entropy(win), 
                             dependencies=['visibility_graph'])
    config.add_record_object("degree_centrality", lambda win: vg_degree_centrality(win), 
                             dependencies=['visibility_graph'])
    config.add_record_object("jaccard_similarity_coefficient", lambda win: vg_jaccard_similarity_coefficient(win), 
                             dependencies=['visibility_graph'])
    config.add_record_object("modularity", lambda win: vg_modularity(win), dependencies=['visibility_graph'])
    config.add_record_object("average_shortest_path_length", lambda win: vg_average_shortest_path_length(win), 
                             dependencies=['visibility_graph'])
    config.add_record_object("closeness_centrality", lambda win: vg_closeness_centrality(win), 
                             dependencies=['visibility_graph'])
    config.add_record_object("small_world", lambda win: vg_small_world(win), 
                             dependencies=['visibility_graph'])
    config.add_record_object("density", lambda win: vg_density(win), 
                             dependencies=['visibility_graph'])
    config.add_record_object("eigenvector_centrality", lambda win: vg_eigenvector_centrality(win), 
                             dependencies=['visibility_graph'])
    config.add_record_object("density", lambda win: vg_density(win), 
                             dependencies=['visibility_graph'])
    config.add_record_object("betweenees_centrality", lambda win: vg_betweenees_centrality(win), 
                             dependencies=['visibility_graph'])
    config.add_record_object("scale_free", lambda win: vg_scale_free(win), 
                             dependencies=['visibility_graph'])
    config.add_record_object("average_clustering_coefficient", lambda win: vg_average_clustering_coefficient(win), 
                             dependencies=['visibility_graph'])
    config.add_record_object("diameter", lambda win: vg_diameter(win), 
                             dependencies=['visibility_graph'])
    config.add_record_object("global_efficiency", lambda win: vg_global_efficiency(win), 
                             dependencies=['visibility_graph'])
    config.add_record_object("local_efficiency", lambda win: vg_local_efficiency(win), 
                             dependencies=['visibility_graph'])
    config.add_record_object("mean_degree", lambda win: vg_mean_degree(win), 
                             dependencies=['visibility_graph'])
    config.add_record_object("transitivity", lambda win: vg_transitivity(win), 
                             dependencies=['visibility_graph'])
    
    
    # Feature on recurrance plot
    config.add_record_object("recurrance_plot", lambda win: calculate_recurrance_plot(win,
        configuration.get("distance_threshold", 0.01)))
    config.add_record_object("recurrence_rate", lambda win: rp_recurrence_rate(win), dependencies=['recurrance_plot'])
    config.add_record_object("determinism", lambda win: rp_determinism(win), dependencies=['recurrance_plot'])
    config.add_record_object("average_diagonal_line_length", lambda win: rp_average_diagonal_line_length(win), dependencies=['recurrance_plot'])
    config.add_record_object("longest_diagonal_line", lambda win: rp_longest_diagonal_line(win), dependencies=['recurrance_plot'])
    config.add_record_object("entropy_of_diagonal_lines", lambda win: rp_entropy_of_diagonal_lines(win), dependencies=['recurrance_plot'])
    config.add_record_object("laminarity", lambda win: rp_laminarity(win), dependencies=['recurrance_plot'])
    config.add_record_object("trapping_time", lambda win: rp_trapping_time(win), dependencies=['recurrance_plot'])
    config.add_record_object("recurrence_time", lambda win: rp_recurrence_time(win), dependencies=['recurrance_plot'])

    # Other features
    config.add_record_object("autocorrelate", lambda win: calculate_auto_corr(win,
        configuration.get("autocorrelation_time_lags", 50)))
    config.add_record_object("fractal_dimension", lambda win: vg_transitivity(win), 
                             dependencies=['fractal_dimension'])
    
    
    return config
