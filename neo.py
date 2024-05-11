import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Texto explicativo sobre o GPT
text = """
O GPT (Generative Pre-trained Transformer) é um modelo de linguagem baseado em inteligência artificial 
que utiliza a arquitetura Transformer. Ele foi desenvolvido pela OpenAI e é conhecido por sua capacidade 
de gerar texto coerente e humano-símile. O GPT é pré-treinado em grandes conjuntos de dados textuais, 
como a Wikipédia, para aprender a estrutura da linguagem humana e capturar padrões de contexto. Após 
o pré-treinamento, o modelo pode ser afinado (fine-tuned) para tarefas específicas, como tradução, 
sumarização de texto, geração de texto e muito mais. O GPT utiliza uma abordagem autoregressiva, o que 
significa que ele gera texto palavra por palavra, tomando como entrada as palavras geradas anteriormente 
para prever a próxima palavra. Isso permite que o modelo produza texto continuamente, criando frases e 
parágrafos coesos e significativos. O GPT tem sido amplamente utilizado em uma variedade de aplicações, 
incluindo assistentes virtuais, geração de texto criativo, completamento automático de texto e muito mais.
"""

# Criar uma lista de palavras únicas
corpus = text.split()

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
def predict_next_word(input_word):
    input_index = word_to_index.get(input_word)
    if input_index is None:
        print(f"A palavra '{input_word}' não está no vocabulário.")
        return None
    input_sequence = np.array([input_index])
    predicted_index = np.argmax(model.predict(input_sequence))
    return index_to_word[predicted_index]

# Função para gerar uma frase utilizando o modelo
def generate_sentence_from_input(user_input, max_length=20):
    sentence = user_input.split()
    current_word = sentence[-1]
    for _ in range(max_length):
        next_word = predict_next_word(current_word)
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
