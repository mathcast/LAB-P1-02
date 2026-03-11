from src.attention import SelfAttention
from src.layernorm import LayerNorm
from src.feedforward import FeedForward

class EncoderLayer:

    def __init__(self, d_model, d_ff):
        self.attention = SelfAttention(d_model)
        self.norm1 = LayerNorm()
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm()
        
    def forward(self, X):
        print("\n--- Encoder Layer ---")
        att = self.attention.forward(X)
        print("After attention:", att.shape)
        X = self.norm1.forward(X + att)
        print("After norm1:", X.shape)
        ffn = self.ffn.forward(X)
        X = self.norm2.forward(X + ffn)
        print("After norm2:", X.shape)
        return X