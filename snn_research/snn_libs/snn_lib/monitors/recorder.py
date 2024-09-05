class NetworkRecorder():
    def __init__(self, item_initial_value, update_function) -> None:
        self.item_initial_value = item_initial_value()
        self.update_function = update_function
        self.record = {
            
        }
        
    def update(self, item_id, value):
        self.record[item_id] = value
        
    def initialize(self, keys):
        for key in keys:
            self.record[key] = self.item_initial_value
            
    def get(self, key):
        return self.record[key]
        
    def __getitem__(self, key):
        return self.record[key]

    def __setitem__(self, key, value):
        self.record[key] = value