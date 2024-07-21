import os
import numpy as np
import tensorflow as tf

BATCH_SIZE = 128
BUFFER_SIZE = 400000
embedding_dim = 1000
rnn_units = 4096
EPOCHS = 75
seq_length = 100

print('BATCH_SIZE:: '+ str(BATCH_SIZE))
print('BUFFER_SIZE:: '+ str(BUFFER_SIZE))
print('embedding_dim:: '+ str(embedding_dim))
print('rnn_units:: '+ str(rnn_units))
print('EPOCHS:: '+ str(EPOCHS))
print('seq_length:: '+ str(seq_length))


# Directory setup
data_dir = './reactjs_docs/'
checkpoint_dir = './training_checkpoints'
model_save_path = './saved_model/model.h5'

# Create directories if they don't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Created directory: {data_dir}")

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    print(f"Created directory: {checkpoint_dir}")

if not os.path.exists(os.path.dirname(model_save_path)):
    os.makedirs(os.path.dirname(model_save_path))
    print(f"Created directory: {os.path.dirname(model_save_path)}")

# 1. Data Collection
texts = []
for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(data_dir, filename), 'r') as file:
            texts.append(file.read())

text = '\n'.join(texts)
print(f'Length of combined text: {len(text)} characters')

# 2. Data Preprocessing
vocab = sorted(set(text))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

examples_per_epoch = len(text) // seq_length

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# 3. Model Design
vocab_size = len(vocab)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size=None):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=False, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    
    if batch_size is not None:
        model.build(tf.TensorShape([batch_size, None]))
    
    return model

# Load the model (if saved model is found)
if os.path.exists(model_save_path):
    model = tf.keras.models.load_model(model_save_path)
    print(f"Loaded model from {model_save_path}")
else:
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=BATCH_SIZE)
    model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy)
    print("No saved model found. Proceeding with a new model.")

# 4. Model Training
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

# Train the model
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# Save the entire model
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# 5. Text Generation and Evaluation
def generate_text(model, start_string):
    num_generate = 1000
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    temperature = 1.0

    # Generate text
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

model.build(tf.TensorShape([1, None]))

# Generate text
test_input = "ReactJS is"
generated_text = generate_text(model, start_string=test_input)
print(generated_text)
