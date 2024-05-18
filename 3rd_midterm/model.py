from keras.models import Sequential, load_model
from keras.layers import GRU, Dropout, TimeDistributed, Dense, Activation, Embedding

def build_model(vocab_size, embedding_dim, rnn_units, num_layers, batch_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                                  output_dim = embedding_dim,
                                  batch_input_shape = [batch_size, None]))
    for i in range(num_layers):
        model.add(GRU(units = rnn_units,
                            return_sequences= True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'))
        
        model.add(Dropout(0.2))
    model.add(Dense(vocab_size))
    
    
    return model