import tensorflow as tf
from typing import List, Union
import numpy as np

class mlp:
    def __init__(self, 
        n_hiddens: List[int], 
        last_activation: str,
        loss_fn,
        optimizer,
        n_features,
        initialization_fn = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        if initialization_fn is None: self.initialization_fn = lambda shape: tf.random.normal(shape=shape)

        n_hiddens.insert(0, n_features)

        self.denses = [
            [
                tf.Variable(self.initialization_fn(shape=(n_hiddens[i], n_hiddens[i+1])), trainable=True),  # Weight
                tf.Variable(tf.zeros(1), trainable=True),  # bias
            ]
            for i in range(len(n_hiddens)-1)
        ]
        self.n_layers = len(self.denses)

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.last_activation = last_activation
    
    def __call__(self, X): # X = 1 minibatch of shape (batch_size, n_features)
        if tf.rank(X) == 1:
            X = tf.expand_dims(X, axis=0)
        
        assert tf.rank(X) == 2

        output = tf.nn.relu( X @ self.denses[0][0] + self.denses[0][1] )

        for i, (W, b) in enumerate(self.denses[1:]):
            if i < self.n_layers-1:
                output = tf.nn.relu(output @ W + b)
            else:
                output = self.last_activation( output @ W + b )
        return output

    @property
    def trainables(self):
        params = []
        for p in self.denses:
            params += [p[0], p[1]]
        return params


    def train(self, X, Y, epoch):
        for e in range(epoch):
            epoch_loss = []
            for x,y in zip(X,Y):
                with tf.GradientTape() as tape:
                    y_pred = self(x)
                    loss = self.loss_fn(y, y_pred)
                
                params = self.trainables

                grads = tape.gradient(loss, params)
                self.optimizer.apply_gradients(zip(grads, params))

                epoch_loss.append(loss.numpy())
            print(f'epoch {e}: loss: {np.array(epoch_loss).mean()}')
            


if __name__ == '__main__':

    n_hiddens = [64, 64, 1]
    last_activation = None
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()


    X = tf.random.normal((10, 80))
    Y = tf.random.normal((10,1))

    
    model = mlp(
        n_hiddens,
        last_activation,
        loss_fn,
        optimizer,
        n_features=X.shape[1]
    )

    out = model(X)
    print(out.shape)

    model.train(X,Y, 20)