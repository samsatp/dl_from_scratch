import tensorflow as tf
import numpy as np
import utils

class RNNModelScratch:
    
    def __init__(self, vocab_size, num_hiddens, vocab: utils.Vocab, initial_state_fn=None):
        
        if initial_state_fn is not None:
            self.initial_state_fn = initial_state_fn
        else:
            self.initial_state_fn = lambda batch_size, num_hiddens: tf.zeros((batch_size, num_hiddens), dtype=tf.float32)

        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.Vocab = vocab
        self.batch_size = None
        
        num_outputs = vocab_size
        
        normal = lambda shape: tf.random.normal(shape=shape, stddev=0.01, mean=0, dtype=tf.float32)
        
        self.W_h = tf.Variable(normal(shape = (vocab_size + num_hiddens, num_hiddens)), dtype=tf.float32)
        self.b_h = tf.Variable(normal(shape = (num_hiddens, )) , dtype=tf.float32)
        self.W_hq= tf.Variable(normal(shape = (num_hiddens, num_outputs)), dtype=tf.float32)
        self.b_q = tf.Variable(normal(shape = (num_outputs, )), dtype=tf.float32)
        
    
    def forward_fn(self, inputs, state):
        """
            inputs: 1 minibatch = (timesteps, batch_size, vocab_size)
            state: (batch_size, num_hiddens)
        """
        outputs = []   # outputs of this batch
        H = state
        self.timesteps = inputs.shape[0]
        self.batch_size = inputs.shape[1]

        for X in inputs:   # for each timestep                 ; X = (batch_size, vocab_size)
            X = tf.concat([X, H], axis=1)                      # X = (batch_size, vocab_size + hidden)
            H = tf.tanh( tf.matmul(X, self.W_h) + self.b_h )   # H = (batch_size, hidden)
            O = tf.matmul(H, self.W_hq) + self.b_q             # O = (batch_size, num_outputs)
            outputs.append(O)
        return tf.reshape(tf.concat(outputs, axis=0), (self.batch_size, self.timesteps, -1)), H                   # y_pred = (timesteps, batch_size, num_outputs)
    
    def __call__(self, X, state):
        return self.forward_fn(X, state)
    
    def begin_state(self, batch_size):
        return self.initial_state_fn(batch_size, self.num_hiddens)
        
    def _get_params(self):
        return [self.W_h, self.b_h, self.W_hq, self.b_q]
    
    @property
    def trainables(self):
        return self._get_params()
        
    @property
    def get_state(self):
        return self.state

    def predict_str(self, prefix, num_preds):
        state = self.begin_state(batch_size=1)

        outputs = [self.Vocab[prefix[0]]]

        get_inputs = lambda : tf.expand_dims(tf.one_hot( [outputs[-1]], self.vocab_size), axis=0)

        for y in prefix[1:]:
            _, state = self.forward_fn(get_inputs(), state)
            outputs.append(self.Vocab[y])

        for _ in range(num_preds):
            y, state = self.forward_fn(get_inputs(), state)
            y_pred = tf.argmax(y, axis=-1)[0].numpy()[0]
            outputs.append(y_pred)
        return outputs

    def train_epoch(self, train_iter, loss_fn, updater):

        def get_inputs(X):
            X = tf.one_hot(X, self.vocab_size)
            X = tf.transpose(X, perm=[1, 0, 2])
            return X

        state = None
        epoch_loss = []
        for X, Y in train_iter: # X and Y : (batch_size, timesteps)
            
            if state is None :
                state = self.begin_state(batch_size=X.shape[0])

            with tf.GradientTape() as tape:
                y_pred, state = self.forward_fn(get_inputs(X), state)  # y_pred : (batch_size, timesteps, vocab_size)
                loss   = loss_fn(Y, y_pred)

            params = self.trainables
            grads  = tape.gradient(loss, params)
            updater.apply_gradients(zip(grads, params))

            epoch_loss.append(loss)
        return sum(epoch_loss)/len(epoch_loss)

    def train(self, train_iter, num_epochs, **kwargs):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        updater = tf.keras.optimizers.Adam()

        losses = []
        for e in range(num_epochs):
            l = self.train_epoch(train_iter, loss_fn, updater)
            losses.append(l)
        
            if (e + 1) % 2 == 0:
                O = self.predict_str('time traveller', 10)
                print(''.join([vocab.idx_to_token[i] for i in O]))
            print(f'{e} loss {sum(losses)/len(losses)}')


if __name__ == "__main__":

    print('===== Getting data')
    lines  = utils.get_time_machine_dataset()
    tokens = utils.tokenize(lines, token='char')
    vocab  = utils.Vocab(tokens)
    print("vocab_size =", len(vocab))


    print('===== Data Loader')
    batch_size, num_steps = 32, 35
    data_iter = utils.DataLoader(batch_size, num_steps, tokens)

    print('===== Model')
    # test model forward pass
    net = RNNModelScratch(len(vocab), num_hiddens=512, vocab=data_iter.vocab)
    initial_state = net.begin_state(batch_size)

    X = tf.random.normal(shape=(num_steps, batch_size, len(vocab)))
    Y, new_state = net(X, initial_state)

    print("PREDICT_STR")
    # test prediction
    O = net.predict_str('time traveller ', 10)
    print(''.join([vocab.idx_to_token[i] for i in O]))

    print('\n\n===== TRAINING\n')
    # transform token into indices
    tokens_indices = [vocab[line] for line in tokens]
    
    train_iter = utils.DataLoader(batch_size, num_steps, tokens_indices)
    net.train(train_iter, 25)


        


