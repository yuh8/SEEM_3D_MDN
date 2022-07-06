import tensorflow as tf
from tensorflow.keras import Model
from .CONSTS import DFF, MAX_NUM_ATOMS, FEATURE_DEPTH, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS


class GraphEmbed(tf.keras.layers.Layer):
    def __init__(self, d_model, rate=0.1):
        super(GraphEmbed, self).__init__()
        self.d_model = d_model
        self.pad = tf.keras.layers.ZeroPadding2D(padding=[0, 1])
        self.dropout = tf.keras.layers.Dropout(rate)

    def build(self, input_shape):
        self.kernel_size = [input_shape[1], 3]
        self.embed = tf.keras.layers.Conv2D(self.d_model,
                                            kernel_size=self.kernel_size,
                                            padding='valid')

    def call(self, x, training):
        '''
        x: [batch_size, num_atoms, num_atoms, feature_depth]
        '''
        # [..., num_atoms, num_atoms + 2, feature_depth]
        x = self.pad(x)
        # [batch_size, num_atoms, d_model]
        x = tf.reduce_sum(self.embed(x), axis=1)
        x = self.dropout(x, training)
        return x


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, num_atoms, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, num_atoms, d_model)
    ])


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.

    Args:
      q: query shape == (..., num_atoms, dk)
      k: key shape == (..., num_atoms, dk)
      v: value shape == (..., num_atoms, dk)
      mask: Float tensor with shape (batch_size, 1, num_atoms) broadcastable
            to (..., num_atoms, num_atoms). Defaults to None


    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., num_atoms, num_atoms)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (num_atoms) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., num_atoms, num_atoms)

    output = tf.matmul(attention_weights, v)  # (..., num_atoms, dk)

    return output, attention_weights


class Sampling(tf.keras.layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.dk = d_model // self.num_heads

        # [num_atoms, d_model]
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, dk).
        Transpose the result such that the shape is (batch_size, num_heads, num_atoms, dk)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.dk))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, v, mask):
        batch_size = tf.shape(x)[0]

        q = self.wq(x)  # (batch_size, num_atoms, d_model)
        k = self.wk(x)  # (batch_size, num_atoms, d_model)
        v = self.wv(v)  # (batch_size, num_atoms, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, num_atoms, dk)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, num_atoms, dk)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, num_atoms, dk)

        # scaled_attention.shape == (batch_size, num_heads, num_atoms, dk)
        # attention_weights.shape == (batch_size, num_heads, num_atoms, num_atoms)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, num_atoms, num_heads, dk)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, num_atoms, d_model)

        output = self.dense(concat_attention)  # (batch_size, num_atoms, d_model)

        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, v, training, mask):

        attn_output, _ = self.mha(x, v, mask)  # (batch_size, num_atoms, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, num_atoms, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, num_atoms, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, num_atoms, d_model)

        return out2


def get_g_net():
    inputs = tf.keras.layers.Input(shape=(MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH))
    mask = tf.reduce_sum(tf.abs(inputs), axis=-1)
    mask = tf.reduce_sum(mask, axis=1, keepdims=True) <= 0
    mask = tf.cast(mask, tf.float32)
    x = GraphEmbed(HIDDEN_SIZE)(inputs)
    for _ in range(NUM_LAYERS):
        # (batch_size, num_atoms, d_model)
        x = EncoderLayer(d_model=HIDDEN_SIZE, num_heads=NUM_HEADS, dff=DFF)(x, x, mask=mask)

    return Model(inputs, x, name='GNet')


def get_gdr_net():
    inputs = tf.keras.layers.Input(shape=(MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 4))
    mask = tf.reduce_sum(tf.abs(inputs), axis=-1)
    mask = tf.reduce_sum(mask, axis=1, keepdims=True) <= 0
    mask = tf.cast(mask, tf.float32)
    x = GraphEmbed(HIDDEN_SIZE)(inputs)
    for _ in range(NUM_LAYERS):
        # (batch_size, num_atoms, d_model)
        x = EncoderLayer(d_model=HIDDEN_SIZE, num_heads=NUM_HEADS, dff=DFF)(x, x, mask=mask)

    # [batch_size, num_atoms, d_model * 2] mean and variance of v to decoder
    z_mean = tf.keras.layers.Dense(HIDDEN_SIZE)(x)
    z_logvar = tf.keras.layers.Dense(HIDDEN_SIZE)(x)
    z = Sampling()((z_mean, z_logvar))
    return Model(inputs, [z_mean, z_logvar, z], name='GDRNet')


def get_decode_net():
    # [batch_size, num_atoms, d_model]
    inputs = tf.keras.layers.Input(shape=(MAX_NUM_ATOMS, HIDDEN_SIZE))
    mask = tf.keras.layers.Input(shape=(1, MAX_NUM_ATOMS))
    v = tf.keras.layers.Input(shape=(MAX_NUM_ATOMS, HIDDEN_SIZE))
    x = EncoderLayer(d_model=HIDDEN_SIZE, num_heads=NUM_HEADS, dff=DFF)(inputs, v, mask=mask)
    for _ in range(NUM_LAYERS - 1):
        # (batch_size, num_atoms, d_model)
        x = EncoderLayer(d_model=HIDDEN_SIZE, num_heads=NUM_HEADS, dff=DFF)(x, x, mask=mask)

    R = tf.keras.layers.Dense(3)(x)
    return Model([inputs, mask, v], R, name='DecoderNet')
