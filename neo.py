import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import re

# Inteligência Artificial (IA) e Seus Conceitos Fundamentais
text = """
## Inteligência Artificial (IA) e Seus Conceitos Fundamentais

A Inteligência Artificial (IA) é um campo da ciência da computação que visa criar sistemas capazes de realizar tarefas que normalmente exigem inteligência humana. Essas tarefas incluem reconhecimento de fala, compreensão de linguagem natural, tomada de decisão e tradução de idiomas. Dentro da IA, existem subcampos específicos como Machine Learning (Aprendizado de Máquina), Deep Learning (Aprendizado Profundo) e Redes Neurais, cada um com suas particularidades e aplicações.

### Inteligência Artificial (IA)
A IA pode ser categorizada em duas principais vertentes: IA Geral (AGI, Artificial General Intelligence) e IA Específica (ANI, Artificial Narrow Intelligence). Enquanto a AGI se refere a sistemas que possuem uma inteligência semelhante à humana em todos os aspectos, a ANI se concentra em resolver problemas específicos.

**Exemplos de Aplicações de IA:**
- **Assistentes Virtuais:** Como Siri, Alexa e Google Assistant.
- **Veículos Autônomos:** Carros que podem dirigir sozinhos.
- **Sistemas de Recomendação:** Algoritmos que sugerem produtos ou conteúdo com base em preferências anteriores, como no Netflix e Amazon.

### Machine Learning (ML)
Machine Learning é um subcampo da IA que envolve o desenvolvimento de algoritmos que permitem aos computadores aprender a partir de dados. Em vez de serem explicitamente programados para realizar uma tarefa, os sistemas de ML são treinados em grandes conjuntos de dados e usam estatísticas para encontrar padrões e fazer previsões.

#### Principais Tipos de Machine Learning:
- **Aprendizado Supervisionado:** O modelo é treinado com dados rotulados. Por exemplo, um sistema que identifica emails de spam é treinado com emails previamente marcados como spam ou não spam.
- **Aprendizado Não Supervisionado:** O modelo trabalha com dados não rotulados e tenta encontrar padrões ou agrupamentos. Por exemplo, segmentação de clientes em marketing.
- **Aprendizado por Reforço:** O modelo aprende através de interações com um ambiente e recebe recompensas ou penalidades. Este tipo é comum em jogos e robótica.

**Exemplo de Algoritmo de ML:**
- **Regressão Linear:** Utilizado para prever valores contínuos, como o preço de uma casa com base em suas características.

### Deep Learning (DL)
Deep Learning é uma subárea do Machine Learning que utiliza Redes Neurais Artificiais com muitas camadas (daí o termo "deep" que significa "profundo") para modelar dados complexos. O DL tem sido responsável por grandes avanços em áreas como visão computacional, processamento de linguagem natural e reconhecimento de fala.

#### Características do Deep Learning:
- **Redes Neurais Convolucionais (CNNs):** Utilizadas principalmente para processamento de imagens e reconhecimento de padrões visuais.
- **Redes Neurais Recorrentes (RNNs):** Utilizadas para dados sequenciais, como texto e séries temporais.
- **Transformers:** Arquitetura avançada para processamento de linguagem natural, como GPT (Generative Pre-trained Transformer).

**Exemplo de Aplicação de DL:**
- **Reconhecimento de Imagens:** Classificação de objetos em imagens, como identificar cães e gatos.

### Redes Neurais Artificiais
Redes Neurais são a base do Deep Learning e são inspiradas no funcionamento do cérebro humano. Uma rede neural é composta por camadas de nós (neurônios artificiais), onde cada nó processa uma parte dos dados de entrada e passa o resultado para a próxima camada.

#### Estrutura de uma Rede Neural:
- **Camada de Entrada:** Onde os dados são introduzidos na rede.
- **Camadas Ocultas:** Onde ocorre o processamento através de nós que aplicam funções matemáticas aos dados.
- **Camada de Saída:** Onde o resultado final é produzido.

### Large Language Models (LLMs)
Large Language Models (Modelos de Linguagem de Grande Escala) são um tipo de modelo de Deep Learning treinado para entender e gerar linguagem natural. Estes modelos são treinados em vastos conjuntos de dados textuais e utilizam arquiteturas como Transformers para capturar o contexto e o significado das palavras.

#### Exemplos de LLMs:
- **GPT-3 e GPT-4:** Modelos desenvolvidos pela OpenAI capazes de gerar texto coerente, traduzir idiomas, responder perguntas e mais.
- **BERT:** Modelo desenvolvido pelo Google que se destaca em tarefas de entendimento de linguagem natural.

**Aplicações de LLMs:**
- **Assistentes de Escrita:** Ajudam na criação de conteúdos textuais.
- **Chatbots:** Fornecem respostas automatizadas a perguntas em interfaces de atendimento ao cliente.
- **Tradução Automática:** Traduzem textos entre diferentes idiomas com alta precisão.

### Conclusão
A inteligência artificial e suas subáreas, incluindo Machine Learning, Deep Learning e Redes Neurais, estão transformando diversos setores ao automatizar e aprimorar tarefas que antes exigiam intervenção humana. Modelos de Linguagem de Grande Escala, como os LLMs, representam um avanço significativo nessa trajetória, permitindo interações mais naturais e eficientes entre humanos e máquinas. A contínua evolução desses campos promete ainda mais inovações e melhorias em nosso dia a dia, aumentando a eficiência e a acessibilidade da tecnologia.
"""

# Função para tokenizar o texto
def tokenize(text):
    text = re.sub(r'\W', ' ', text.lower())  # Remover caracteres não alfanuméricos e converter para minúsculas
    tokens = text.split()
    return tokens

# Tokenizar o texto
corpus = tokenize(text)

# Criar pares de entrada e saída para o modelo
input_sequences = []
next_words = []
for i in range(len(corpus) - 1):
    input_sequences.append(corpus[i])
    next_words.append(corpus[i + 1])

# Criar dicionários para mapear palavras para índices e vice-versa
word_to_index = {word: i for i, word in enumerate(set(corpus))}
index_to_word = {i: word for word, i in word_to_index.items()}

# Preparar os dados de entrada e saída para o modelo
X = np.array([word_to_index[word] for word in input_sequences])
y = np.array([word_to_index[word] for word in next_words])

# Redimensionar os dados de entrada para serem tridimensionais
X = np.expand_dims(X, axis=-1)

# Construir o modelo
vocab_size = len(word_to_index)
embedding_dim = 50
model = Sequential([
    Embedding(vocab_size, embedding_dim),
    LSTM(100),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinar o modelo
model.fit(X, y, epochs=100)

# Função para prever a próxima palavra
def predict_next_word(input_word, temperature=1.0):
    input_index = word_to_index.get(input_word)
    if input_index is None:
        print(f"A palavra '{input_word}' não está no vocabulário.")
        return None
    input_sequence = np.array([input_index])
    predictions = model.predict(input_sequence, verbose=0)[0]
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-7) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    predicted_index = np.argmax(probas)
    return index_to_word[predicted_index]

# Função para gerar uma frase utilizando o modelo
def generate_sentence_from_input(user_input, max_length=20, temperature=1.0):
    sentence = user_input.lower().split()
    current_word = sentence[-1]
    for _ in range(max_length):
        next_word = predict_next_word(current_word, temperature=temperature)
        if next_word is None:
            break
        sentence.append(next_word)
        if next_word in ('.', '?', '!'):
            break
        current_word = next_word
    return ' '.join(sentence)

# Ponto de entrada
if __name__ == "__main__":
    while True:
        user_input = input("Digite uma palavra ou frase: ")
        if user_input.lower() == 'sair':
            print("Saindo...")
            break
        generated_sentence = generate_sentence_from_input(user_input)
        print("Frase gerada:", generated_sentence)
