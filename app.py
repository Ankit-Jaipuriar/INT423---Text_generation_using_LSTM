import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import re

# Load your trained model
model = tf.keras.models.load_model('my_text_generation_model_v2.keras')

# Load the vocab_mapping from token.pkl
with open('vocab_mapping.pkl', 'rb') as f:
    vocab_mapping = pickle.load(f)

# Function to convert integers back to text
def convert_int_to_text(int_tokens, vocab_mapping):
    reverse_vocab = {v: k for k, v in vocab_mapping.items()}
    return [reverse_vocab.get(i, "<UNK>") for i in int_tokens]

def preprocess_input_text(input_text):
    input_text = input_text.strip().lower()
    input_text = re.sub(r'[^a-z\s]', '', input_text)
    return input_text

def sample_next_word(predictions, temperature):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-7) / temperature
    predictions = np.exp(predictions) / np.sum(np.exp(predictions))  # Softmax
    return np.random.choice(len(predictions[0]), p=predictions[0])

def predict_next_word(input_text, temperature=0.5):
    input_text = preprocess_input_text(input_text)
    word_tokens = input_text.split()
    
    # Ensure all tokens are mapped and handle unknown tokens with vocab.get()
    int_tokens = [vocab_mapping.get(token, vocab_mapping.get("<UNK>")) for token in word_tokens]
    
    # Convert to numpy array of integers and ensure it's 2D
    int_tokens = np.array(int_tokens, dtype=np.int32).reshape(1, -1)
    
    # Predict the next word
    prediction = model.predict(int_tokens)
    
    # Sample the next word
    prediction_idx = sample_next_word(prediction, temperature)
    
    # Return the predicted word
    return convert_int_to_text([prediction_idx], vocab_mapping)[0]

def generate_text(input_text, n_words, context_length=30):
    word_sequence = input_text.split()
    context = word_sequence[:]
    
    for _ in range(n_words):
        prediction = predict_next_word(' '.join(context), temperature=0.7)
        word_sequence.append(prediction)
        context.append(prediction)
        
        if len(context) > context_length:  # Limit context to the last 30 words
            context.pop(0)

    generated_text = ' '.join(word_sequence)
    generated_text = generated_text.capitalize()
    
    if not generated_text.endswith(('.', '!', '?')):  # Ensure it ends with punctuation
        generated_text += '.'

    return generated_text

# Streamlit UI
st.title("LSTM Text Generator")
input_text = st.text_input("Enter the initial text:", "Tom got one look at Huck")
n_words = st.slider("Number of words to generate:", 5, 100, 50)  # Allow up to 100 words

if st.button("Generate"):
    generated_text = generate_text(input_text, n_words)
    st.write(generated_text)
