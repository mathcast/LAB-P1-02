# LAB P1-02: Implementação de um Transformer Encoder com NumPy

## Como baixar o projeto

```bash
git clone https://github.com/mathcast/LAB-P1-02.git
cd transformer-encoder-lab
```

## Fórmulas implementadas

### Self-Attention

```
Attention(Q, K, V) = softmax((Q K^T) / √d_k) V
```

* **Q (Query)**, **K (Key)** e **V (Value)** são projeções da entrada X.
* Calcula-se o **produto escalar** entre Q e K^T.
* O resultado é **dividido por √d_k** (fator de escala).
* Aplica-se **softmax em cada linha**, gerando pesos de atenção.
* Os **pesos são multiplicados por V**, produzindo a saída da atenção.

---

### Feed Forward Network

```
FFN(x) = max(0, xW1 + b1)W2 + b2
```

* Primeira multiplicação linear: `xW1 + b1`
* Aplicação da função **ReLU**
* Segunda multiplicação linear: `W2 + b2`
* Cada token é processado **independentemente**

---

### Layer Normalization

```
LayerNorm(x) = (x - média) / √(variância + ε)
```

* Calcula a **média e variância por token**
* Normaliza os valores do vetor
* Mantém a estabilidade numérica da rede

---

### Estrutura de uma Encoder Layer

```
Self Attention
↓
Add + LayerNorm
↓
Feed Forward
↓
Add + LayerNorm
```

---

## Estrutura do repositório

```
transformer-encoder-lab/
├── src/
│   ├── embeddings.py      # Criação de embeddings para os tokens
│   ├── attention.py       # Implementação do Self-Attention
│   ├── layernorm.py       # Implementação do Layer Normalization
│   ├── feedforward.py     # Implementação da Feed Forward Network
│   ├── encoder_layer.py   # Uma camada completa do Transformer Encoder
│   └── transformer.py     # Empilhamento de múltiplas camadas do encoder
│
├── data/
│   └── sentences.txt      # Dataset de frases utilizado como entrada
│
├── main.py                # Executa todo o pipeline do modelo
├── requirements.txt       # Dependências do projeto
└── README.md
```

---

## Pipeline do modelo

O fluxo executado pelo `main.py` é:

```
Frases do dataset
↓
Leitura do arquivo sentences.txt
↓
Criação automática do vocabulário
↓
Tokenização das palavras
↓
Conversão para IDs numéricos
↓
Embeddings
↓
Transformer Encoder (6 camadas)
↓
Saída contextualizada
```

---

## Como rodar

### 1. Ambiente virtual e dependências

No diretório do projeto:

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Executar o projeto

```bash
python main.py
```

O script irá:

1. Ler o dataset (`sentences.txt`)
2. Criar o vocabulário automaticamente
3. Escolher uma frase aleatória
4. Converter palavras em tokens
5. Gerar embeddings
6. Processar os dados através do Transformer Encoder
7. Exibir o vetor de saída contextualizado

---

### 3. Exemplo de saída

Durante a execução, o programa mostra cada etapa do processo:

```
ETAPA 1: LENDO DATASET
ETAPA 2: CRIANDO VOCABULÁRIO
ETAPA 3: ESCOLHENDO FRASE
ETAPA 4: TOKENIZAÇÃO
ETAPA 5: EMBEDDINGS
ETAPA 6: CRIANDO TRANSFORMER
ETAPA 7: PROCESSANDO NO TRANSFORMER
```

Saída final:

```
Shape da saída: (1, 4, 64)
```

Onde:

* **1** → batch size
* **4** → número de tokens da frase
* **64** → dimensão dos embeddings

---

## Como usar no seu código

Exemplo simples de uso do encoder:

```python
from src.transformer import TransformerEncoder
from src.embeddings import Embeddings
import numpy as np

vocab_size = 20
d_model = 64
d_ff = 128

embedding = Embeddings(vocab_size, d_model)

tokens = np.array([[1,2,3,4]])

X = embedding.forward(tokens)

encoder = TransformerEncoder(
    num_layers=6,
    d_model=d_model,
    d_ff=d_ff
)

output = encoder.forward(X)
```

---

## Dataset

O dataset utilizado está em:

```
data/sentences.txt
```

Exemplo de frases:

```
o banco bloqueou cartao
o cliente pagou conta
o banco aprovou credito
```

Cada linha representa **uma frase utilizada como entrada do modelo**.

---

## Requisitos técnicos

* **Linguagem:** Python 3
* **Dependência:** apenas NumPy
