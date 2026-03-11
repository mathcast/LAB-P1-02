import numpy as np

class FeedForward:

    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model)
        self.b2 = np.zeros(d_model)
        
    def forward(self, X):
        print("\nFeedForward input:", X.shape)
        h = X @ self.W1 + self.b1
        print("After first layer:", h.shape)
        h = np.maximum(0, h)
        out = h @ self.W2 + self.b2
        print("After second layer:", out.shape)
        return out