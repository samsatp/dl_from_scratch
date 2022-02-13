import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

class Additive_attn(tf.keras.layers.Layer):
    """
        attention_weights = softmax(a(q, k))  ;a is a scoring function
        a(q,k) = Wv tanh( Wq (q) + Wk (k) ) ∈ ℝ

        attention_pooling_output = a(q, k) @ V

        query: 𝐪∈ℝ𝑞, key: 𝐤∈ℝ𝑘
        trainables: 𝐖𝑞∈ℝℎ×𝑞 , 𝐖𝑘∈ℝℎ×𝑘, and 𝐰𝑣∈ℝℎ.
    """
    def __init__(
        self,
        num_hiddens, 
        dropout, 
        **kwargs
        ) -> None:
        super().__init__(**kwargs)

        self.Wq = Dense(units=num_hiddens, use_bias=False)
        self.Wk = Dense(units=num_hiddens, use_bias=False)
        self.Wv = Dense(units=1, use_bias=False)
        self.dropout = Dropout(dropout)

        self.num_hiddens = num_hiddens

    def masked_softmax(self, X, valid_len):
        if valid_len is None: return X
        # mask unused
        seq_mask_bool = tf.sequence_mask(valid_len, maxlen=X.shape[-1])
        seq_mask_float = tf.cast(seq_mask_bool, dtype=tf.float32)

        # reshape mask to the same as X
        seq_mask_float = tf.reshape(seq_mask_float, shape=X.shape)
        seq_mask_bool = tf.reshape(seq_mask_bool, shape=X.shape)

        # multiply X by mask
        X = tf.multiply(tf.cast(X, dtype=tf.float32), seq_mask_float)
        
        # change masked from 0.0 to -1e6
        value_mask = tf.where(seq_mask_bool, 0.0, -1e6)
        X_masked = X + value_mask

        return tf.nn.softmax(X_masked, axis=-1)

    def call(self, Q, K, V, valid_len, **kwargs):
        """
            Q: (, 1, dec_emb_size)
            K: (, enc_steps, enc_n_hiddens) ;same as K
        """
        Q = self.Wq(Q)  # (, 1, num_hiddens)
        K = self.Wk(K)  # (, enc_steps, num_hiddens)

        attn_weights = self.Wv(tf.nn.tanh(Q + K)) # (, enc_steps, num_hiddens) @ (, 1) = (, enc_steps, 1)

        attn_weights = tf.transpose(attn_weights, perm=[0,2,1]) # (, 1, enc_steps)
        self.attention_weights = self.masked_softmax(attn_weights, valid_len) # (, 1, enc_steps)

        attn_pooling_output = self.dropout(self.attention_weights) @ V # (, 1, enc_steps) @ (, enc_steps, enc_num_hiddens) = (, 1, enc_num_hiddens)

        return attn_pooling_output



if __name__ == '__main__':
    import sys
    query = sys.argv[1]
    print(query)

    if query == 'call':
        batch_size = 3
        n_queries = 1
        dec_emb_size = 10

        enc_steps = 15
        enc_num_hiddens = 64

        Q = tf.random.normal(shape=(batch_size, n_queries, dec_emb_size))
        K = V = tf.random.normal(shape=(batch_size, enc_steps, enc_num_hiddens))

        valid_lens = tf.constant([2, 6, 10])

        attn = Additive_attn(num_hiddens=9, dropout=0.5)

        outputs = attn(Q, K, V, valid_lens)
        print("attn weights =",attn.attention_weights.shape)
        print("pooling outputs =",outputs.shape)


