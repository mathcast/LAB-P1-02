from src.encoder_layer import EncoderLayer

class TransformerEncoder:

    def __init__(self, num_layers, d_model, d_ff):
        self.layers = [
            EncoderLayer(d_model, d_ff)
            for _ in range(num_layers)
        ]
        
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X