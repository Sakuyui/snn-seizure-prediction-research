import numpy as np
import os, sys

from scripts.dynamical_feature_extraction.feature_extractor import RecordConfiguration
from scripts.dynamical_feature_extraction.features_func import *

class BaseRecorderBuilder():
    def __init__(self):
        self.config = RecordConfiguration()

    def build_recorder_config(self, n_channels):
        raise NotImplementedError
        

class DynamicalFeatureRecorderBuilder(BaseRecorderBuilder):
    def __init__(self):
        super().__init__()
    
    def build_recorder_config(self, n_channels):
        config : RecordConfiguration = self.config
        configuration = {}
        # # Dynamic network-based features
        config.add_record_object("electrode_sd", electrode_sd)
        config.add_record_object("electrode_correlation", electrode_correlation)
        config.add_record_object("mutual_information_2d", mutual_information_2d, inject_arguments={
            'time_step_lags': 10
        }, inject_padding={'padding_left': 200}, temporary=True)
        
        config.add_record_object("mutual_information_flatten", mutual_information_flatten, dependencies=['mutual_information_2d'])
        
        config.add_record_object("distribution_entropy", distribution_entropy, dependencies=['mutual_information_flatten'])
        config.add_record_object("network_entropy", network_entropy, dependencies=['mutual_information_2d'])
        
        config.add_record_object("conditional_transfer_information", conditional_transfer_information,
            inject_padding={
                'padding_left': 200
            }, inject_arguments={'time_step_lags': 10}, use_pca=True)
        
        config.add_record_object("visibility_graph", visibility_graph, apply_per_channel=True, use_pca=True)
        
        config.add_record_object("vg_degrees", vg_degrees, dependencies=['visibility_graph'], apply_per_channel=True, use_pca=True)
        config.add_record_object("vg_degree_local_entropy", vg_degree_local_entropy, dependencies=['vg_degrees', 'visibility_graph'], apply_per_channel=True, use_pca=True)
        config.add_record_object("vg_degree_distribution", vg_degree_distribution, dependencies=['vg_degrees'], apply_per_channel=True, use_pca=True)
        config.add_record_object("vg_degree_entropy", vg_degree_entropy, dependencies=['vg_degree_distribution'], apply_per_channel=True, use_pca=True)
        config.add_record_object("vg_degree_centrality", vg_degree_centrality, dependencies=['visibility_graph'], apply_per_channel=True, use_pca=True) # check!
        config.add_record_object("vg_graph_index_complexity", vg_graph_index_complexity, dependencies=['visibility_graph'], apply_per_channel=True, use_pca=True)
        config.add_record_object("vg_jaccard_similarity_coefficient", vg_jaccard_similarity_coefficient,
                                 dependencies=['visibility_graph'], apply_per_channel=True, use_pca=True)
        config.add_record_object("vg_density", vg_density,
                                 dependencies=['visibility_graph'], apply_per_channel=True, use_pca=True)

        config.add_record_object("vg_average_shortest_path_length", vg_average_shortest_path_length,
                                 dependencies=['visibility_graph', 'full_source_full_destination_shortest_path'], apply_per_channel=True, use_pca=True)
        config.add_record_object("full_source_full_destination_shortest_path", full_source_full_destination_shortest_path,
                                 dependencies=['visibility_graph'], apply_per_channel=True, use_pca=True)
        config.add_record_object("vg_closeness_centrality", vg_closeness_centrality,
                                 dependencies=['full_source_full_destination_shortest_path'], apply_per_channel=True, use_pca=True)
        # config.add_record_object("vg_betweenees_centrality", vg_betweenees_centrality,
        #                          dependencies=['visibility_graph'], apply_per_channel=True)

        config.add_record_object("vg_diameter", vg_diameter,
                                 dependencies=['full_source_full_destination_shortest_path'], apply_per_channel=True, use_pca=True)
        config.add_record_object("vg_global_efficiencies", vg_global_efficiency,
                                 dependencies=['full_source_full_destination_shortest_path', 'visibility_graph'], apply_per_channel=True, use_pca=True)
        
        config.add_record_object("vg_local_efficiency", vg_local_efficiency,
                                 dependencies=['full_source_full_destination_shortest_path'], apply_per_channel=True, use_pca=True)
        config.add_record_object("vg_mean_degree", vg_mean_degree,
                                 dependencies=['vg_degrees'], apply_per_channel=True, use_pca=True)
        config.add_record_object("vg_average_weighted_degree", vg_average_weighted_degree,
                                 dependencies=['visibility_graph'], apply_per_channel=True, use_pca=True)
        config.add_record_object("vg_transitivity", vg_transitivity,
                                 dependencies=['visibility_graph'], apply_per_channel=True, use_pca=True)
        
        config.add_record_object("reconstructed_phase_space", reconstructed_phase_space,
                                 dependencies=[], inject_padding={'padding_left': 200}, inject_arguments={
                                     'delay_time_steps': 10
                                 })
        
        config.add_record_object("recurrence_plot", recurrence_plot,
                                 dependencies=['reconstructed_phase_space'], inject_arguments={
                                     'distance_threshold': configuration.get('distance_threshold', 0.2)
                                 })
       
        config.add_record_object("rp_recurrence_rate", rp_recurrence_rate, 
                                 dependencies=['recurrence_plot'])
        config.add_record_object("diagnoal_length_distribution", diagnoal_length_distribution,
                                 dependencies=['recurrence_plot'])
        config.add_record_object("vertical_length_distribution", vertical_length_distribution,
                                 dependencies=['recurrence_plot'])
        
        config.add_record_object("rp_determinism", rp_determinism, 
                                 dependencies=['diagnoal_length_distribution'], inject_arguments={
                                     'l_min': configuration.get("l_min", 2)
                                 })
        config.add_record_object("rp_average_diagonal_line_length", rp_average_diagonal_line_length,
                                 dependencies=['diagnoal_length_distribution'], inject_arguments={
                                    'l_min': configuration.get("l_min", 2)
                                 })
        
        config.add_record_object("rp_longest_diagonal_line", rp_longest_diagonal_line, 
                                 dependencies=['diagnoal_length_distribution'])
        config.add_record_object("rp_entropy_of_diagonal_lines", rp_entropy_of_diagonal_lines,
                                 dependencies=['diagnoal_length_distribution'])
        config.add_record_object("rp_laminarity", rp_laminarity,
                                 dependencies=['vertical_length_distribution'], inject_arguments={
                                     'l_min': configuration.get("l_min", 2)
                                 })
        config.add_record_object("rp_trapping_time", rp_trapping_time,
                                 dependencies=['diagnoal_length_distribution'], inject_arguments={
                                     'l_min': configuration.get("l_min", 2)
                                 })
        
        config.add_record_object("katz_fractal_dimensions", katz_fractal_dimensions)
        
        return self.config