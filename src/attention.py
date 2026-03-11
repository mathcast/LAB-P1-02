import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

class SelfAttention:
    
    def __init__(self, d_model):
        self.Wq = np.random.randn(d_model, d_model)
        self.Wk = np.random.randn(d_model, d_model)
        self.Wv = np.random.randn(d_model, d_model)
        self.dk = d_model

    def forward(self, X):
        print("\nSelf Attention Input shape:", X.shape)
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv

        print("Q shape:", Q.shape)
        print("K shape:", K.shape)
        print("V shape:", V.shape)
        scores = Q @ K.transpose(0,2,1)
        print("Scores shape:", scores.shape)
        scores = scores / np.sqrt(self.dk)
        attn = softmax(scores)
        print("Attention matrix:")
        print(attn)
        output = attn @ V
        return output