import tensorflow as tf
import numpy as np
from model import build_model
checkpoint_dir = 'training_checkpoints_2_200'

data = open('data/review.txt', 'r').read()
vocab = sorted(list(set(data)))
char2idx = {char:i for i, char in enumerate(vocab)}
idx2char = np.array(vocab)

# Length of vocabulary in chars
vocab_size = len(vocab)
# The embedding dimension
embedding_dim = 256
# Number of RNN units
rnn_units = 200

def generate(start_string, out_len, temperature, rnn_units, num_layers):

    data = open('data/review.txt', 'r').read()
    vocab = sorted(list(set(data)))
    char2idx = {char:i for i, char in enumerate(vocab)}
    idx2char = np.array(vocab)

    checkpoint_dir = f'training_checkpoints_{num_layers}_{rnn_units}'
    # Length of vocabulary in chars
    vocab_size = len(vocab)
    # The embedding dimension
    embedding_dim = 256

    tf.train.latest_checkpoint(checkpoint_dir)
    model = build_model(vocab_size, embedding_dim, rnn_units, num_layers,batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    # Builds the model based on input shapes received.
    model.build(tf.TensorShape([1, None]))
    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    # convert (x,y) shaped matrix to (1,x,y).
    input_eval = tf.expand_dims(input_eval, axis=0) 
    
    # Empty string to store our results
    text_generated = []
    
    # Here batch size == 1
    model.reset_states()
    for i in range(out_len):
        predictions = model(input_eval)
        
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)
        
        # using a categorical distribution to predict the 
        # character returned by the model
        predictions = predictions / temperature
        
        # We got the predictions for every timestep but we 
        # want only last so first we take [-1] to consider on last 
        # predictions distribution only and after we try to get id 
        # from 1D array. Ex. we got '2' from a=['2'] by a[0].
        predicted_id = tf.random.categorical(predictions, 
                                             num_samples=1
                                            )[-1,0].numpy()
        
        # We pass the predicted character as the next input to the 
        # model along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
        
        text_generated.append(idx2char[predicted_id])
        
    return (start_string + ''.join(text_generated)).rsplit(' ', 1)[0]  # Drops the last word (often truncated)

if __name__ == '__main__':
    
    print(generate(checkpoint_dir, 'it is unacceptable', 500, 0.))