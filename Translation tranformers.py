import pandas as pd
import re
from unicodedata import normalize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from keras_nlp.layers import TokenAndPositionEmbedding, TransformerEncoder, TransformerDecoder

def load_and_process_data(file_path, sequence_len=None):
    # Load the dataset
    df = pd.read_csv('/content/sample_data/en-fr.txt', names=['en', 'fr', 'attr'], usecols=['en', 'fr'], sep='\t')
    
    # Shuffle and reset index
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Clean the text
    def clean_text(text):
        text = normalize('NFD', text.lower())
        text = re.sub('[^A-Za-z ]+', '', text)
        return text

    def clean_and_prepare_text(text):
        text = '[start] ' + clean_text(text) + ' [end]'
        return text

    df['en'] = df['en'].apply(clean_text)
    df['fr'] = df['fr'].apply(clean_and_prepare_text)

    # Calculate max lengths if not provided
    if sequence_len is None:
        en_max_len = max(len(line.split()) for line in df['en'])
        fr_max_len = max(len(line.split()) for line in df['fr'])
        sequence_len = max(en_max_len, fr_max_len)

    print(f'Max phrase length (English): {en_max_len}')
    print(f'Max phrase length (French): {fr_max_len}')
    print(f'Sequence length: {sequence_len}')

    # Fit Tokenizers and generate padded sequences
    en_tokenizer = Tokenizer()
    en_tokenizer.fit_on_texts(df['en'])
    en_sequences = en_tokenizer.texts_to_sequences(df['en'])
    en_x = pad_sequences(en_sequences, maxlen=sequence_len, padding='post')

    fr_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n')
    fr_tokenizer.fit_on_texts(df['fr'])
    fr_sequences = fr_tokenizer.texts_to_sequences(df['fr'])
    fr_y = pad_sequences(fr_sequences, maxlen=sequence_len + 1, padding='post')

    return en_x, fr_y, en_tokenizer, fr_tokenizer, sequence_len

def build_transformer_model(en_vocab_size, fr_vocab_size, sequence_len):
    num_heads = 8
    embed_dim = 256

    # Encoder
    encoder_input = Input(shape=(None,), dtype='int64', name='encoder_input')
    x = TokenAndPositionEmbedding(en_vocab_size, sequence_len, embed_dim)(encoder_input)
    encoder_output = TransformerEncoder(embed_dim, num_heads)(x)

    # Decoder
    decoder_input = Input(shape=(None,), dtype='int64', name='decoder_input')
    x = TokenAndPositionEmbedding(fr_vocab_size, sequence_len, embed_dim, mask_zero=True)(decoder_input)
    x = TransformerDecoder(embed_dim, num_heads)(x, encoder_output)
    x = Dropout(0.4)(x)

    # Output layer
    decoder_output = Dense(fr_vocab_size, activation='softmax')(x)

    # Define the model
    model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary(line_length=120)

    return model

# Example usage
file_path = 'Data/en-fr.txt'
en_x, fr_y, en_tokenizer, fr_tokenizer, sequence_len = load_and_process_data(file_path)

# Build the Transformer model
en_vocab_size = len(en_tokenizer.word_index) + 1
fr_vocab_size = len(fr_tokenizer.word_index) + 1
transformer_model = build_transformer_model(en_vocab_size, fr_vocab_size, sequence_len)

# Now you can train the model
# transformer_model.fit([en_x, fr_y[:, :-1]], fr_y[:, 1:], epochs=10, batch_size=64)  # Uncomment and adjust as needed
