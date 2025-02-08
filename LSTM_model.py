import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Flatten
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform

# Self Attention Layer
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W_q = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer=GlorotUniform(), trainable=True)
        self.W_k = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer=GlorotUniform(), trainable=True)
        self.W_v = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer=GlorotUniform(), trainable=True)
        super(SelfAttention, self).build(input_shape)
    
    def call(self, inputs):
        Q = tf.matmul(inputs, self.W_q)
        K = tf.matmul(inputs, self.W_k)
        V = tf.matmul(inputs, self.W_v)
        attention_scores = tf.nn.softmax(tf.matmul(Q, K, transpose_b=True) / tf.sqrt(tf.cast(K.shape[-1], tf.float32)))
        return tf.matmul(attention_scores, V)

# Model construction
def build_LSTM(input_shape):
    input_layer = Input(shape=input_shape)
    lstm_out = LSTM(50, return_sequences=True)(input_layer)
    attention_out = SelfAttention()(lstm_out)
    flatten_out = Flatten()(attention_out)

    output_layer = Dense(1, activation='linear')(flatten_out)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    
    return model
