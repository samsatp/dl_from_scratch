import tensorflow as tf
from typing import List, Union
from abc import ABC, abstractmethod
import utils

############## INTERFACES ##############
class Encoder(tf.keras.layers.Layer, ABC):

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    @abstractmethod
    def call(self, X, *args, **kwargs):
        raise NotImplementedError

class Decoder(tf.keras.layers.Layer, ABC):

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @abstractmethod
    def init_state(self, encoder_states):
        raise NotImplementedError

    @abstractmethod
    def call(self, X, *args, **kwargs):
        raise NotImplementedError

class EncoderDecoder(tf.keras.Model, ABC):

    def __init__(self, encoder, decoder, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, enc_X, dec_X):

        _, states = self.encoder(enc_X)
        dec_state = self.decoder.init_state(states)
        return self.decoder(dec_X, dec_state)

    @abstractmethod
    def predict(self, src_sentence, num_steps, src_vocab, tgt_vocab):
        raise NotImplementedError

    @abstractmethod
    def train(self, data_iter, epochs, **kwargs):
        raise NotImplementedError

#############################################

class seq2seq_Encoder(Encoder):
    def __init__(self, n_layers, n_hiddens, emb_dims, vocab_size, dropout=0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.embedding = tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = emb_dims)
        self.rnn = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells([
                tf.keras.layers.GRUCell(units = n_hiddens, dropout = dropout)
                for _ in range(n_layers)
            ],
            name = 'encoder_rnn'
            ),
            return_state = True,
            return_sequences = True
        )

    def call(self, X):
        assert tf.rank(X) == 2
        batch_size, timesteps = X.shape

        embeded = self.embedding(X)  # (, timesteps, emb_dims)
        sequences, *states = self.rnn(embeded)

        return sequences, states

class seq2seq_Decoder(Decoder):
    def __init__(self, n_layers, n_hiddens, emb_dims, vocab_size, dropout=0.0, **kwargs) -> None:
        super().__init__(**kwargs)

        self.embedding = tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = emb_dims)
        self.rnn = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells([
                tf.keras.layers.GRUCell(units = n_hiddens, dropout=dropout)
                for _ in range(n_layers)
            ]),
            return_state = True,
            return_sequences = True
        )
        self.dense = tf.keras.layers.Dense(units = vocab_size)

    def init_state(self, encoder_states):
        return encoder_states

    def call(self, X, states, *args, **kwargs):

        # X: (, timesteps) & states: (, enc_layers, enc_n_hiddens)

        X = self.embedding(X) # (, timesteps, emb_dims)
        context = states[-1]  # (, enc_n_hiddens)
        context = tf.expand_dims(context, axis=1)                   # (, 1, enc_n_hiddens)
        context = tf.repeat(context, repeats=X.shape[1], axis=1)    # (, timesteps, enc_n_hiddens)

        dec_input = tf.concat([X, context], axis=2)                 # (, timesteps, enc_n_hiddens + emb_dims)

        sequences, *states = self.rnn(dec_input, initial_state=states)
        output = self.dense(sequences)

        return output, states
        
class seq2seq(EncoderDecoder):
    def __init__(self, encoder, decoder, **kwargs) -> None:
        super().__init__(encoder, decoder, **kwargs)

    def predict(self, 
        src_sentence: List[List[str]],  # [['t', 'i', 'm']]
        num_steps: int, 
        src_vocab: utils.Vocab, 
        tgt_vocab: utils.Vocab
    ) -> str:

        enc_X = tf.constant(src_vocab[src_sentence])   # enc_X: [[5, 3, 13]]        
        dec_X = tf.constant(tgt_vocab[ [['<bos>']] ])  # dec: [[99]]
        
        _, states = self.encoder(enc_X)

        output_indices = ''.join(src_sentence[0])
        eos_index = tgt_vocab['<eos>']
        for _ in range(num_steps):
            Y_pred, states = self.decoder(dec_X, states)
            Y_pred = tf.argmax(Y_pred, axis=-1)            # predicted index
            dec_X = Y_pred

            Y_pred = tf.reshape(Y_pred, shape=()).numpy()
            if Y_pred == eos_index: break
            Y_pred = tgt_vocab.to_tokens( int(Y_pred) )    # convert index to token
            output_indices += Y_pred


        return output_indices

    def train(self, data_iter, epochs, **kwargs):
        return 


def sequence_mask(X, valid_len):
    if valid_len is None: return X
    mask = tf.sequence_mask(valid_len, maxlen=X.shape[-1])
    X = tf.cast(X, tf.float32)
    mask = tf.cast(mask, tf.float32)

    X = tf.multiply(X, mask)
    return X


class MaskedSoftmaxCELoss(tf.keras.losses.Loss):
    def __init__(self, valid_len):
        # valid_len: (, )
        super().__init__(reduction='none')
        self.valid_len = valid_len

    def call(self, label, pred):
        # pred: (, timesteps, vocab_size)
        # label:(, timesteps)
        weights = tf.ones_like(label, dtype=tf.float32)
        weights = sequence_mask(weights, self.valid_len)
        
        label_one_hot = tf.one_hot(label, depth=pred.shape[-1])
        unweighted_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')(label_one_hot, pred)
        weighted_loss = tf.reduce_mean((unweighted_loss*weights), axis=1)
        return weighted_loss




if __name__ == '__main__':
    import sys
    query = sys.argv[1]
    print(query)

    if query == 'enc_dec':
        print('='*20)
        # Get data
        lines  = utils.get_time_machine_dataset()
        tokens = utils.tokenize(lines, token='char')
        vocab  = utils.Vocab(tokens)

        # Model
        encoder = seq2seq_Encoder(vocab_size=len(vocab), emb_dims=64, n_hiddens=16, n_layers=2)
        decoder = seq2seq_Decoder(vocab_size=len(vocab), emb_dims=64, n_hiddens=16, n_layers=2)
        model = seq2seq(encoder, decoder)

        # Prediction
        outputs = model.predict(utils.tokenize(["time traveller "]), 20, vocab, vocab)
        print(outputs)
    
    elif query == 'mask_seq':
        X = tf.constant([[1, 2, 3], [4, 5, 6]])
        print(sequence_mask(X, tf.constant([1, 2])))

    elif query == 'masked_loss':
        loss = MaskedSoftmaxCELoss(tf.constant([4, 2, 0]))
        y_true = tf.ones((3,4))
        y_pred = tf.ones((3, 4, 10))

        print(loss(y_true, y_pred).numpy())


