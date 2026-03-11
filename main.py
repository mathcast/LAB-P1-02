import numpy as np
import random
from src.embeddings import Embeddings
from src.transformer import TransformerEncoder

print("\n==============================")
print("TRANSFORMER ENCODER LAB")
print("==============================")
print("\n--- ETAPA 1: LENDO DATASET ---")

with open("data/sentences.txt", "r") as f:
    lines = f.readlines()
sentences = [line.strip().split() for line in lines]

print("Sentenças carregadas:")

for s in sentences:
    print(s)

print("\n--- ETAPA 2: CRIANDO VOCABULÁRIO ---")

vocab = {}
idx = 0
for sentence in sentences:
    for word in sentence:
        if word not in vocab:
            vocab[word] = idx
            idx += 1

print("Vocabulário:")
print(vocab)
vocab_size = len(vocab)
print("\n--- ETAPA 3: ESCOLHENDO FRASE ---")
sentence = random.choice(sentences)
print("Frase escolhida:")
print(sentence)
print("\n--- ETAPA 4: TOKENIZAÇÃO ---")
ids = np.array([[vocab[w] for w in sentence]])
print("Tokens numéricos:")
print(ids)
print("\n--- ETAPA 5: EMBEDDINGS ---")
d_model = 64
d_ff = 128
embedding = Embeddings(vocab_size, d_model)
X = embedding.forward(ids)
print("Shape embeddings:", X.shape)
print("Primeiro vetor embedding:")
print(X[0][0])
print("\n--- ETAPA 6: CRIANDO TRANSFORMER ---")

encoder = TransformerEncoder(
    num_layers=6,
    d_model=d_model,
    d_ff=d_ff
)

print("Encoder criado com 6 camadas")
print("\n--- ETAPA 7: PROCESSANDO NO TRANSFORMER ---")
output = encoder.forward(X)
print("\n--- RESULTADO FINAL ---")
print("Shape da saída:")
print(output.shape)
print("\nPrimeiro vetor contextualizado:")
print(output[0][0])
print("\nProcesso finalizado.")