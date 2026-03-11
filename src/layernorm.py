import numpy as np

class LayerNorm:

    def __init__(self, eps=1e-6):
        self.eps = eps
        
    def forward(self, X):
        mean = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        norm = (X - mean) / np.sqrt(var + self.eps)
        return norm