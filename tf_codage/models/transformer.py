"""
Tensorflow implementation of Transformer model:

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, `Attention is all you need`_, 2017

Adapted from tensorflow `tutorial`_.

.. _tutorial: https://github.com/tensorflow/docs/blob/e876022a05aaabfd2340bf84fff1464f14429377/site/en/tutorials/text/transformer.ipynb
.. _Attention is all you need: https://arxiv.org/pdf/1706.03762.pdf
"""

from dataclasses import dataclass, asdict
import json
from typing import Optional
import numpy as np
import tensorflow as tf


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq, pad_token_id=0):
    seq = tf.cast(tf.math.equal(seq, pad_token_id), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1
    )  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(dff, activation="relu"),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model),  # (batch_size, seq_len, d_model)
        ]
    )


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # (batch_size, input_seq_len, d_model)
        return out2


class Encoder(tf.keras.layers.Layer):
    """Encoder part of the transformer architecture."""

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        maximum_position_encoding,
        rate=0.1,
        input_vocab_size=None,
    ):
        """Creates an encoder layer.

        Encoder layer takes a sequence of token ids (or embeddings, if `input_vocab_size` is None)
        and returns a sequence of contextualised embeddings.

        Args:
            num_layers: number of encoder layers
            d_model: number of units in hidden layers (and embedding size)
            num_heads: number of attention heads
            dff: number of units in feedforward layers
            maximum_position_encoding: maximum input sequence length
            rate: dropout rate
            input_vocab_size: size of input vocabulary
              (or None in which case the inputs must be
              already embeddings).
        """
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        if input_vocab_size:
            self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        else:
            self.embedding = None
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """Run the encoder layer with the inputs.

        Args:
            x: sequence of token ids or embeddings
              (if `input_vocabulary_size` is None)
            training: use True if used during training and False 
              for evaluation
            mask: mask for padding tokens
        """

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        if self.embedding:
            x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(
            x, x, x, look_ahead_mask
        )  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask
        )  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(
            ffn_output + out2
        )  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
    """Decoder part of the transformer architecture."""

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        target_vocab_size,
        maximum_position_encoding,
        rate=0.1,
    ):
        """Initialize a decoder.

        Args:
            num_layers: number of encoder layers
            d_model: number of units in hidden layers (and embedding size)
            num_heads: number of attention heads
            dff: number of units in feedforward layers
            target_vocab_size: size of the vocabulary
            maximum_position_encoding: maximum input sequence length
            rate: dropout rate
        """

        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask
            )

            attention_weights["decoder_layer{}_block1".format(i + 1)] = block1
            attention_weights["decoder_layer{}_block2".format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    """Complete transformer model"""

    def __init__(self, config):
        """Creates a new transformer.

        Creates a complete encoder-decoder architecture for transformer.
        It can accept both the embeddings or learn the embeddings on its
        own.

          >>> n_tokens = 5
          >>> vocab_size = 10
          >>> pad_token_id = 0
          >>> config = TransformerConfig(
          ...     pe_input=n_tokens,
          ...     pe_target=n_tokens,
          ...     decoder_pad_token_id=pad_token_id,
          ...     target_vocab_size=vocab_size,
          ...     input_vocab_size=vocab_size)
          >>> transformer = Transformer(config)
          >>> transformer({
          ...     "input_ids": np.array([[2, 3, 4]]),
          ...     "attention_mask": np.array([[0, 0, 1]]),
          ...     "codes": np.array([[1, pad_token_id, pad_token_id]]),
          ... }, training=False)
          <tf.Tensor: shape=(1, 3, 10), dtype=float32, numpy=...>
        
        Args:
            config: model parameters as a TransformerConfig object

        Returns:
            an instance of transformer model.

        """
        super(Transformer, self).__init__()
        self.decoder_pad_token_id = config.decoder_pad_token_id

        self.encoder = Encoder(
            config.num_layers,
            config.d_model,
            config.num_heads,
            config.dff,
            config.pe_input,
            config.rate,
            input_vocab_size=config.input_vocab_size,
        )

        # flag to check if user will pass embeddings or token ids
        self._token_inputs = bool(config.input_vocab_size)

        self.decoder = Decoder(
            config.num_layers,
            config.d_model,
            config.num_heads,
            config.dff,
            config.target_vocab_size,
            config.pe_target,
            config.rate,
        )

        self.final_layer = tf.keras.layers.Dense(config.target_vocab_size)

    def call(self, inputs, training=None):
        """
        Inputs is a dictionary with the following keys:
        
        - input_ids - token ids for the input sequence
        - input_embeds - embeddings for the input sequence (if input_ids not given)
        - codes - inputs for the decoder (normally the same as target but shifted by one token)
        - attention_mask - mask to mark padding tokens (1 for padding token, 0 for data)
        """

        if self._token_inputs:
            inp = inputs["input_ids"]
        else:
            inp = inputs["inputs_embeds"]
        tar = inputs["codes"]
        enc_padding_mask = tf.cast(1 - inputs["attention_mask"], tf.float32)[
            :, None, None, :
        ]
        dec_padding_mask = enc_padding_mask

        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])

        ##  add masking non-code padded outputs
        dec_target_padding_mask = create_padding_mask(tar, self.decoder_pad_token_id)
        look_ahead_mask = tf.maximum(look_ahead_mask, dec_target_padding_mask)

        enc_output = self.encoder(
            inp, training, enc_padding_mask
        )  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask
        )

        final_output = self.final_layer(
            dec_output
        )  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output


@dataclass
class TransformerConfig:
    """Common parameters for transformer model (Transformer class):
    
    Args:
        num_layer: number of layers
        d_model: size of embedding vector and hidden layers
        num_heads:  number of attention heads
        dff:  number of units in feedforward sublayers
        target_vocab_size: number of symbols in target dictionary
        pe_input: number of position embeddings (tokens) in the encoder input
        pe_target: number of position embeddings in the decoder output
        rate: dropout rate
        decoder_pad_token_id: token number used for padding decoder inputs
        input_vocab_size: size of input vocabulary
          if input_vocab_size = None the inputs have to be already embeddings
          (as `input_embeds` key in input dictionary).

    Default parameters are taken from base transformer model of the paper:

    Vaswani et al. "Attention is all you need", arXiv: 1706.03762, https://arxiv.org/pdf/1706.03762.pdf
    """

    num_layers: int = 6
    d_model: int = 512
    num_heads: int = 8
    dff: int = 2048

    # these values need to be overridden
    target_vocab_size: int = 0
    pe_input: int = 0
    pe_target: int = 0

    rate: float = 0.1
    decoder_pad_token_id: Optional[int] = None
    input_vocab_size: Optional[int] = None

    def __post_init__(self):
        self.num_layers = int(self.num_layers)
        if self.decoder_pad_token_id:
            self.decoder_pad_token_id = int(self.decoder_pad_token_id)

    def as_json(self):
        """dump the configuration to json string"""
        return json.dumps(asdict(self), sort_keys=True, indent=4)

    def save(self, file_path):
        """save the model in a json file"""
        with open(file_path, "wt") as fid:
            fid.write(self.as_json())

    @classmethod
    def load(cls, file_path):
        """load the model from a json file:

        Args:
            file_path: path to the file

        Returns: config object
        """
        with open(file_path, "rt") as fid:
            restored_params = json.load(fid)
        return cls(**restored_params)
