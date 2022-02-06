import tensorflow as tf
from typing import List, Union
import numpy as np

class mlp(tf.keras.Model):
    def __init__(self, 
        n_hiddens: List[int], 
        last_activation: str,
        loss_fn,
        optimizer,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        last_layer = n_hiddens.pop(-1)
        self.denses = [
            tf.keras.layers.Dense(units = n, activation='relu')
            for n in n_hiddens
        ]

        self.denses.append(
            tf.keras.layers.Dense(last_layer, activation = last_activation)
        )

        self.loss_fn = loss_fn
        self.optimizer = optimizer
    
    def call(self, X):
        X = tf.expand_dims(X, axis=0)
        output = self.denses[0](X)
        for dense in self.denses[1:]:
            output = dense( output )
        return output

    @property
    def trainables(self):
        _params = [e.trainable_variables for e in self.denses]
        params = []
        for p in _params:
            params += [p[0], p[1]]
        return params


    def train(self, X, Y, epoch):
        for e in range(epoch):
            epoch_loss = []
            for x,y in zip(X,Y):
                with tf.GradientTape() as tape:
                    y_pred = self.call(x)
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

    model = mlp(
        n_hiddens,
        last_activation,
        loss_fn,
        optimizer
    )

    X = tf.random.normal((10, 80))
    Y = tf.random.normal((10,1))

    out = model(X)
    print(out.shape)

    model.train(X,Y, 20)