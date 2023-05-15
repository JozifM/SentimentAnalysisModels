import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFAutoModel

# Load and preprocess the data
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin', engine='python', header=None)
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
df = df.sample(n=50000)

X = df['text']
labels = df['target']
labels[labels == 4] = 1

# Tokenize and pad the sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)

max_len = 50
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# Convert labels to one-hot encoded format
labels = tf.keras.utils.to_categorical(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Create the attention mask matrix
attention_mask = np.ones_like(X_train)
padded_positions = np.equal(X_train, 0)
attention_mask[padded_positions] = 0

# Define the BERT model
bert = TFAutoModel.from_pretrained('bert-base-cased')

# Build the model
input_seq = tf.keras.layers.Input(shape=(max_len,), name='sequence', dtype='int32')
mask = tf.keras.layers.Input(shape=(max_len,), name='mask', dtype='int32')

embeddings = bert(input_seq, attention_mask=mask)[0]

X = tf.keras.layers.MaxPooling1D()(embeddings)
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Dense(128, activation='relu')(X)
X = tf.keras.layers.Dropout(0.1)(X)
X = tf.keras.layers.Dense(32, activation='relu')(X)
y = tf.keras.layers.Dense(2, activation='sigmoid', name='output')(X)  # Two output neurons for binary classification

model = tf.keras.Model(inputs=[input_seq, mask], outputs=y)

# Compile the model
optimizer = tf.keras.optimizers.Adam(0.01)
loss = tf.keras.losses.BinaryCrossentropy()
acc = tf.keras.metrics.BinaryAccuracy(name='accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

# Train the model
history = model.fit(
    x=[X_train, attention_mask],
    y=y_train,
    validation_data=([X_test, attention_mask], y_test),
    epochs=3,
    batch_size=32
)
