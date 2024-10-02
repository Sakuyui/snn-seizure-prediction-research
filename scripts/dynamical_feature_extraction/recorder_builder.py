from scripts.dynamical_feature_extraction.feature_extractor import *

class RecorderBuilder():
    def __init__(self):
        pass
    
    def _build_record_path(self, recorder_configuration, parallelism_detection=True):
        
        key_list = list(recorder_configuration)
        configuration_id_map = {key:index for index, key in enumerate(key_list)}
        n = len(configuration_id_map)

        dependency_graph = np.zeros((n, n))
        for i in range(n):
            dependencies = recorder_configuration[key_list[i]]['dependencies']
            for dep in dependencies:
                dependency_graph[configuration_id_map[dep], i] = 1

        path = []
        processed_node_id = set()
        cnt_nodes = 0

        # Topological sort.
        while cnt_nodes < n:
            processed = False
            if parallelism_detection:
                parallelizable_node_ids = []
                for i in range(n):
                    enter_edge_vector = dependency_graph[:, i]
                    if i in processed_node_id:
                        continue                    
                    if np.all(enter_edge_vector == 0):
                        parallelizable_node_ids.append(i)
                        processed_node_id.add(i)
                        cnt_nodes += 1
                        processed = True
                if parallelizable_node_ids:
                    dependency_graph[parallelizable_node_ids, :] = 0
                    print(f'recognized parallelizable nodes: {parallelizable_node_ids}, {[key_list[parallelizable_node_id] for parallelizable_node_id in parallelizable_node_ids]}')
                    path.append([key_list[node_id] for node_id in parallelizable_node_ids])
            else:
                for i in range(n):
                    enter_edge_vector = dependency_graph[:, i]
                    if i in processed_node_id:
                        continue
                    if np.all(enter_edge_vector == 0):
                        dependency_graph[i, :] = 0
                        path.append(key_list[i])
                        processed_node_id.add(i)
                        cnt_nodes += 1
                        processed = True
                        break

            if not processed:
                raise ArithmeticError("Cyclic dependency detected in the configuration.")
        
        return path
    
    
    def build_record(self, record_objects = {}, enable_pca = -1, parallelism_detection=True, separate_parallelizable_records = False):
        record = FeatureRecorder(record_objects)
        
        if enable_pca > 0:
            record.ipca = IncrementalPCA(n_components=enable_pca)
            record.enable_pca = True
        else:
            record.enable_pca = False
            
        record_path = self._build_record_path(recorder_configuration=record_objects, parallelism_detection=parallelism_detection)
        record.record_path = record_path
        record.prepared = True
        if parallelism_detection:
            if not separate_parallelizable_records:
                return record
            return [FeatureRecorder(RecordConfiguration(record_objects={parallelizable_function: record_objects[parallelizable_function] for parallelizable_function in parallelizable_functions}), record_path=parallelizable_functions).use_pca(4) for parallelizable_functions in record_path]
        else:
            return record
    
