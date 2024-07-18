import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import gensim

# Texto de treinamento (corpus)
text = """
## Inteligência Artificial (IA) e Seus Conceitos Fundamentais
...
"""

# Função para tokenizar o texto
def tokenize(text):
    text = re.sub(r'\W', ' ', text.lower())  # Remover caracteres não alfanuméricos e converter para minúsculas
    tokens = text.split()
    return tokens

# Função para preparar os dados para treinamento
def prepare_data(corpus):
    input_sequences = []
    next_words = []
    for i in range(len(corpus) - 1):
        input_sequences.append(corpus[i])
        next_words.append(corpus[i + 1])

    word_to_index = {word: i for i, word in enumerate(set(corpus))}
    index_to_word = {i: word for word, i in word_to_index.items()}

    X = np.array([word_to_index[word] for word in input_sequences])
    y = np.array([word_to_index[word] for word in next_words])
    X = np.expand_dims(X, axis=-1)

    return X, y, word_to_index, index_to_word

# Tokenizar o texto
corpus = tokenize(text)

# Preparar os dados
X, y, word_to_index, index_to_word = prepare_data(corpus)

# Estratégia 1: Modelo base com treinamento simples
def train_simple_model(X, y, vocab_size, embedding_dim=50):
    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        LSTM(100),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=10)
    return model

# Estratégia 2: Modelo com embeddings pré-treinados
def load_pretrained_embeddings(word_to_index, embedding_dim=50):
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format('path/to/word2vec.bin', binary=True)
    embedding_matrix = np.zeros((len(word_to_index), embedding_dim))
    for word, index in word_to_index.items():
        if word in word_vectors:
            embedding_matrix[index] = word_vectors[word]
    return embedding_matrix

def train_pretrained_embedding_model(X, y, vocab_size, embedding_matrix):
    embedding_dim = embedding_matrix.shape[1]
    model = Sequential([
        Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False),
        LSTM(100),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=10)
    return model

# Estratégia 3: Uso de GPT-2 sem ajuste fino
def generate_text_gpt2(prompt, max_length=50):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Estratégia 4: Ajuste fino com GPT-2 (esboço simplificado)
# Para um ajuste fino real, você precisa de mais recursos computacionais e um dataset apropriado
def train_gpt2(dataset_path, model_name='gpt2', output_dir='./gpt2-finetuned'):
    from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=dataset_path,
        block_size=128
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(output_dir)

# Estratégia 5: Geração controlada com temperatura
def predict_next_word(model, input_word, word_to_index, index_to_word, temperature=1.0):
    if input_word not in word_to_index:
        print(f"A palavra '{input_word}' não está no vocabulário. Usando a palavra mais próxima.")
        input_word = find_closest_word(input_word)
        print(f"Palavra mais próxima encontrada: '{input_word}'")
        
    input_index = word_to_index[input_word]
    input_sequence = np.array([input_index])
    predictions = model.predict(input_sequence, verbose=0)[0]
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-7) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    predicted_index = np.argmax(probas)
    return index_to_word[predicted_index]

def generate_sentence_from_input(model, user_input, word_to_index, index_to_word, max_length=20, temperature=1.0):
    sentence = user_input.lower().split()
    current_word = sentence[-1]
    for _ in range(max_length):
        next_word = predict_next_word(model, current_word, word_to_index, index_to_word, temperature=temperature)
        if next_word is None:
            break
        sentence.append(next_word)
        if next_word in ('.', '?', '!'):
            break
        current_word = next_word
    return ' '.join(sentence)

# Estratégia 6: Geração com beam search
def generate_text_beam_search(model, tokenizer, prompt, beam_width=3, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_beams=beam_width, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Ponto de entrada principal
if __name__ == "__main__":
    vocab_size = len(word_to_index)
    
    # Treinar modelos
    simple_model = train_simple_model(X, y, vocab_size)
    
    # Carregar embeddings pré-treinados
    # Certifique-se de fornecer o caminho correto para o arquivo de embeddings pré-treinados
    embedding_matrix = load_pretrained_embeddings(word_to_index)
    pretrained_embedding_model = train_pretrained_embedding_model(X, y, vocab_size, embedding_matrix)
    
    # GPT-2
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Geração de frases
    user_input = "Digite uma palavra ou frase: "
    
    print("Estratégia 1: Modelo base")
    for _ in range(5):
        print(generate_sentence_from_input(simple_model, user_input, word_to_index, index_to_word))
    
    print("Estratégia 2: Modelo com embeddings pré-treinados")
    for _ in range(5):
        print(generate_sentence_from_input(pretrained_embedding_model, user_input, word_to_index, index_to_word))
    
    print("Estratégia 3: Uso de GPT-2 sem ajuste fino")
    for _ in range(5):
        print(generate_text_gpt2(user_input))
    
    # Ajuste fino com GPT-2: Esta parte requer um dataset e recursos computacionais adequados
    # train_gpt2('path/to/your/training_corpus.txt')
    # gpt2_finetuned_model = GPT2LMHeadModel.from_pretrained('./gpt2-finetuned')
    
    print("Estratégia 5: Geração controlada com temperatura")
    for _ in range(5):
        print(generate_sentence_from_input(pretrained_embedding_model, user_input, word_to_index, index_to_word, temperature=0.7))
    
    print("Estratégia 6: Geração com beam search")
    for _ in range(5):
        print(generate_text_beam_search(gpt2_model, gpt2_tokenizer, user_input))
