class CKYBacktrackingContext():
    def __init__(self):
        self._context = {}
        
    def __getitem__(self, index):
        return self._context[index]
    def __setitem__(self, key, value):
        self._context[key] = value