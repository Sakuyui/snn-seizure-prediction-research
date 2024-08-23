import numpy as np
import os, sys
sys.path.append(".")

from feature_extractor import RecordConfiguration
from features_func import *

def RecorderBuilder():
    
    def __init__(self):
        self.config = RecordConfiguration()

    def build_recorder_config(self, n_channels, configuration={}):
        config : RecordConfiguration = self.config
        
        # Dynamic network-based features
        config.add_record_object("electrode_correlation", electrode_correlation)
        config.add_record_object("mutual_information_2d", mutual_information_2d, inject_arguments={
            'time_step_lags': 10
        }, inject_padding={'padding_left':10}, temporary=True)
        config.add_record_object("mutual_information_flatten", mutual_information_flatten, dependencies=['mutual_information_2d'])
        config.add_record_object("distribution_entropy", distribution_entropy, dependencies=['mutual_information'])
        config.add_record_object("network_entropy", network_entropy, dependencies=['mutual_information_2d'])
        config.add_record_object("visibility_graph", visibility_graph)
        config.add_record_object("conditional_transfer_information", conditional_transfer_information, inject_padding=True,
                                 inject_arguments={'time_step_lags': 10}, apply_per_channel=True)
        config.add_record_object("vg_degrees", vg_degree, dependencies=['visibility_graph'], apply_per_channel=True)
        config.add_record_object("vg_degree_local_entropy", vg_degree_local_entropy, dependencies=['vg_degree', 'visibility_graph'])
        config.add_record_object("vg_degree_distribution", vg_degree_distribution, dependencies=['vg_degree'], apply_per_channel=True)
        config.add_record_object("vg_degree_entropy", vg_degree_entropy, dependencies=['vg_degree_distribution'], apply_per_channel=True)
        config.add_record_object("vg_degree_centralitie", vg_degree_centrality, dependencies=['vg_degree'], apply_per_channel=True)
        config.add_record_object("vg_graph_index_complexity", vg_graph_index_complexities, dependencies=['visibility_graph'], apply_per_channel=True)
        config.add_record_object("vg_jaccard_similarity_coefficient", vg_jaccard_similarity_coefficient,
                                 dependencies=['visibility_graph'], apply_per_channel=True)
        config.add_record_object("vg_density", vg_density,
                                 dependencies=['visibility_graph'])
        config.add_record_object("vg_average_shortest_path_length", vg_average_shortest_path_length,
                                 dependencies=['visibility_graph', 'full_source_full_destination_shortest_path'])
        config.add_record_object("vg_closeness_centrality", vg_closeness_centrality,
                                 dependencies=['full_source_full_destination_shortest_path'])
        config.add_record_object("vg_betweenees_centrality", vg_betweenees_centrality,
                                 dependencies=['visibility_graph'])
        config.add_record_object("vg_diameter", vg_diameter,
                                 dependencies=['full_source_full_destination_shortest_path'])
        config.add_record_object("vg_global_efficiencies", vg_global_efficiency,
                                 dependencies=['full_source_full_destination_shortest_path'])
        config.add_record_object("vg_local_efficiency", vg_local_efficiency,
                                 dependencies=['full_source_full_destination_shortest_path'])
        config.add_record_object("vg_mean_degree", vg_mean_degree,
                                 dependencies=['vg_degree'])
        config.add_record_object("vg_avergae_weighted_degree", vg_avergae_weighted_degree,
                                 dependencies=['vg_avergae_weighted_degree'])
        config.add_record_object("vg_transitivity", vg_transitivity,
                                 dependencies=['visibility_graph'])
        config.add_record_object("full_source_full_destination_shortest_path", full_source_full_destination_shortest_path,
                                 dependencies=['visibility_graph'])
        config.add_record_object("reconstructed_phase_space", full_source_full_destination_shortest_path,
                                 dependencies=['visibility_graph'], apply_per_channel=True)
        
        config.add_record_object("recurrance_plot", recurrance_plot,
                                 dependencies=['reconstructed_phase_space'], inject_arguments={
                                     'distance_threshold': configuration.get('distance_threshold', 0.2)
                                 }, apply_per_channel=True)
        config.add_record_object("recurrance_plot", recurrance_plot,
                                 dependencies=['reconstructed_phase_space'], inject_arguments={
                                     'distance_threshold': configuration.get('distance_threshold', 0.2)
                                 }, apply_per_channel=True)
        config.add_record_object("rp_recurrence_rate", rp_recurrence_rate, 
                                 dependencies=['recurrance_plot'], apply_per_channel=True) # TODO: check dependencies's `apply_per_channel`
        config.add_record_object("diagnoal_length_distribution", diagnoal_length_distribution,
                                 dependencies=['recurrance_plot'], apply_per_channel=True)
        config.add_record_object("vertical_length_distribution", vertical_length_distribution,
                                 dependencies=['recurrance_plot'], apply_per_channel=True)
        config.add_record_object("rp_determinism", rp_determinism, 
                                 dependencies=['diagnoal_length_distribution'], inject_arguments={
                                     'l_min': config.get("l_min", 2)
                                 }, apply_per_channel=True)
        config.add_record_object("rp_average_diagonal_line_length", rp_average_diagonal_line_length,
                                 dependencies=['diagnoal_length_distribution'], inject_arguments={
                                    'l_min': config.get("l_min", 2)
                                 }, apply_per_channel=True)
        config.add_record_object("rp_longest_diagonal_line", rp_longest_diagonal_line, 
                                 dependencies=['diagnoal_length_distribution'], apply_per_channel=True)
        config.add_record_object("rp_entropy_of_diagonal_lines", rp_entropy_of_diagonal_lines,
                                 dependencies=['diagnoal_length_distribution'], apply_per_channel=True)
        config.add_record_object("rp_laminarity", rp_laminarity,
                                 dependencies=['vertical_length_distribution'], inject_arguments={
                                     'l_min': config.get("l_min", 2)
                                 },apply_per_channel=True)
        config.add_record_object("rp_trapping_time", rp_trapping_time,
                                 dependencies=['length_distribution'], inject_arguments={
                                     'l_min': config.get("l_min", 2)
                                 },apply_per_channel=True)
        
        config.add_record_object("katz_fractal_dimensions", katz_fractal_dimensions)
        config.add_record_object("electrode_sd", electrode_sd)
        
        return self.config
