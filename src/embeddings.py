import numpy as np

class Embeddings:

    def __init__(self, vocab_size, d_model):
        self.embedding = np.random.randn(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding[x]