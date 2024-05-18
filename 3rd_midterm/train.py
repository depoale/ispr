import tensorflow as tf
import numpy as np
import os
import time
from model import build_model

DATA_PATH = 'data/review.txt'
BATCH_SIZE = 100
BUFFER_SIZE = 10000

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def train(text=DATA_PATH):
    data = open('data/review.txt', 'r').read()
    vocab = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(vocab)
    print("----------------------------------------")
    print("Data has {} characters, {} unique".format(data_size, vocab_size))
    print("----------------------------------------")
    char2idx = {char:i for i, char in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_encoded = [char2idx[c] for c in data]
    text_encoded = np.array(text_encoded)

    # Show how the first 31 characters from the text are mapped to integers
    print('Text: {} \n==> Encoded as : {}'.format(text[:31], text_encoded[:31]))

    # The maximum length sentence we want for a single input in characters
    seq_length = 150
    example_per_epoch = len(data)//seq_length # as we have 1 example of seq_length characters.
    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_encoded)
    for i in char_dataset.take(5):
        print(idx2char[i.numpy()])
    
    sequences = char_dataset.batch(batch_size=seq_length+1, drop_remainder=True)
    # repr function print string representation of an object.
    for item in sequences.take(5):
        pass
        #print(repr(''.join(idx2char[item.numpy()])))

    dataset = sequences.map(split_input_target)
    for input_ex, target_ex in dataset.take(1):
        print('Input data: ', repr(''.join(idx2char[input_ex.numpy()])))
        print('Output data:',repr(''.join(idx2char[target_ex.numpy()])))


    for i, (input_idx, target_idx) in enumerate(zip(input_ex[:5], target_ex[:5])):
        print("Step {:4d}".format(i))
        print("Input : {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
        print("Expected output : {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    print(dataset)

    # Length of vocabulary in chars
    vocab_size = len(vocab)
    # The embedding dimension
    embedding_dim = 256
    # Number of RNN units
    rnn_units = 200
    # Number of layers
    num_layers=3

    model = build_model(vocab_size = len(vocab),
                    embedding_dim = embedding_dim,
                    rnn_units = rnn_units,
                    num_layers=num_layers,
                    batch_size = BATCH_SIZE)
    model.summary()

    # First check the shape of the output:
    for input_ex_batch, target_ex_batch in dataset.take(1):
        ex_batch_prediction = model(input_ex_batch) # simply it takes input and calculate output with initial weights.
        print(ex_batch_prediction.shape, "# (batch_size, sequence_length, vocab_size)")

    sampled_indices = tf.random.categorical(ex_batch_prediction[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    print(sampled_indices)

    print('Input: \n', repr("".join(idx2char[input_ex_batch[0].numpy()])))
    print('\nPredicted sequence for next characters is: \n', repr("".join(idx2char[sampled_indices])))

    ex_batch_loss = loss(target_ex_batch, ex_batch_prediction)
    print("Prediction shape: ", ex_batch_prediction.shape, " # (batch_size, sequence_length, vocab_size)")
    print("Scaler loss: ", ex_batch_loss.numpy().mean())

    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = f'./training_checkpoints_{num_layers}_{rnn_units}'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_prefix, save_weights_only=True)
    EPOCHS = 10
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

if __name__ == '__main__':
    train()