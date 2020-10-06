"""
Implement models for various tasks. Most of the models
are sub-classed of the huggingface transformers models, so please
checkout their  `docs <https://huggingface.co/transformers/>`_ for more information
"""

import os
import GPUtil
from transformers import TFBertForSequenceClassification, BertConfig
from transformers import CamembertTokenizer
from transformers import TFRobertaForSequenceClassification
from transformers import TFRobertaModel
from transformers import CamembertConfig
import tensorflow as tf


class TFCamembertForSequenceClassification(TFRobertaForSequenceClassification):
    config_class = CamembertConfig
    pretrained_model_archive_map = {}


class TFCamembertModel(TFRobertaModel):
    config_class = CamembertConfig
    pretrained_model_archive_map = {}


class TFCamembertForSentenceEmbedding(TFCamembertModel):
    """Camembert model for calculating sentence embeddings through an average of
    word embeddings.
    """

    def call(self, inputs, *args, **kwargs):

        word_embeddings, _ = super().call(inputs, *args, **kwargs)
        if isinstance(inputs, dict) and "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"]
            mask = tf.expand_dims(tf.cast(attention_mask, dtype=tf.float32), axis=2)
            sentence_embeddings = tf.reduce_sum(
                word_embeddings * mask, axis=1
            ) / tf.reduce_sum(mask, axis=1)
        else:
            sentence_embeddings = tf.reduce_mean(word_embeddings, axis=1)

        return sentence_embeddings

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_trackable_saver")
        return state


class FullTextBert(TFCamembertModel):
    """Camembert model that can handle long documents by splitting them into separate token batches.

    The model produces contextualised embeddings and needs to be paired with a task head 
    (for classification etc.) such as ``PoolingClassificationHead``.

    Creating a new model from scratch (only context embeddings, without task head):

      >>> tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
      >>> cls_token = tokenizer.cls_token_id
      >>> sep_token = tokenizer.sep_token_id
      >>> config = FullTextConfig()
      >>> model = FullTextBert(config, cls_token=cls_token, sep_token=sep_token)
      >>> inputs = {
      ...     "input_ids": tf.constant([[1, 2, 3]]),
      ...     "attention_mask": tf.constant([[0, 0, 0]])}
      >>> embeddings = model(inputs)


    """

    def __init__(
        self,
        config,
        *inputs,
        cls_token=None,
        sep_token=None,
        max_batches=10,
        layer_id=-1,
        **kwargs
    ):
        """
        Initialize a FullTextBert model with given parameters.
                  
        Args:
           config: instance ``FullTextConfig`` with model parameters
           cls_token: token id corresponding to the CLS token
           sep_token: token id corresponding to the SEP token
           max_batches: maximum number of sequences to split the document into.
           layer_id: which layer to use in the output, layer_id=0 is the first layer,
             layer_id=-1 (the default) is the last layer
        """

        if layer_id != -1:
            config.output_hidden_states = True

        self.layer_id = layer_id

        super().__init__(config, *inputs, **kwargs)
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.max_batches = max_batches

        # we need to compute new mask and propagate it
        self._compute_output_and_mask_jointly = True

    @property
    def input_size(self):
        return self.config.max_position_embeddings - 2

    def pad_mask(self, attention_mask):

        max_batches = self.max_batches
        input_size = self.input_size
        bert_seq_len = input_size - 2
        _, seq_len = attention_mask.shape

        padded_mask = tf.pad(
            attention_mask,
            [[0, 0], [0, max_batches * bert_seq_len - seq_len]],
            constant_values=0,
        )
        attention_mask = tf.reshape(padded_mask, [-1, bert_seq_len])
        attention_mask = tf.pad(attention_mask, [[0, 0], [1, 1]], constant_values=1)
        return attention_mask

    def call(self, inputs, mask=None, **kwargs):
        """Evalulate the model for inputs.

        Args:
           inputs: hugging-face like inputs (normally dict with keys
             ``input_ids``, ``token_type_ids``, ``attention_mask``

        Returns:
           contextualised embeddings
        """

        cls_token = self.cls_token
        sep_token = self.sep_token

        max_batches = self.max_batches
        input_size = self.input_size
        hidden_size = self.config.hidden_size

        # take account of special tokens
        bert_seq_len = input_size - 2

        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", mask)
        else:
            input_ids = inputs
            attention_mask = mask

        n_seq, seq_len = input_ids.shape
        padded_inputs = tf.pad(
            input_ids,
            [[0, 0], [0, max_batches * bert_seq_len - seq_len]],
            constant_values=1,
        )

        new_tensor = tf.reshape(padded_inputs, [-1, bert_seq_len])

        padded_tensor = tf.pad(new_tensor, [[0, 0], [1, 0]], constant_values=cls_token)
        padded_tensor = tf.pad(
            padded_tensor, [[0, 0], [0, 1]], constant_values=sep_token
        )

        if attention_mask is not None:
            attention_mask = self.pad_mask(attention_mask)

        outputs = self.roberta(padded_tensor, attention_mask=attention_mask, **kwargs)

        if self.layer_id == -1:
            sequence_output = outputs[0]
        else:
            sequence_output = outputs[2][self.layer_id]

        reshaped_outputs = tf.reshape(
            sequence_output, [-1, max_batches, input_size, hidden_size]
        )

        if attention_mask is not None and self._compute_output_and_mask_jointly:
            reshaped_outputs._keras_mask = tf.reshape(
                attention_mask, [-1, max_batches, input_size]
            )

        return reshaped_outputs


class BertForMultilabelClassification(TFBertForSequenceClassification):
    """Bert for sequence classification with sigmoid activation in the output layer"""

    def call(self, inputs, **kwargs):
        outputs = super().call(inputs, **kwargs)
        activation = tf.keras.layers.Activation("sigmoid")
        probs = activation(outputs[0])
        return probs


class CamembertForMultilabelClassification(TFCamembertForSequenceClassification):
    """Camembert for sequence classification with sigmoid activation in the output layer"""

    def call(self, inputs, **kwargs):
        outputs = super().call(inputs, **kwargs)
        activation = tf.keras.layers.Activation("sigmoid")
        probs = activation(outputs[0])
        return probs


class MeanMaskedPooling(tf.keras.layers.Layer):
    """Mean pooling with strides and mask"""

    def __init__(self, config, *args, **kwargs):
        pool_size = config.pool_size
        strides = config.pool_strides
        num_labels = config.num_labels

        super().__init__(*args, **kwargs)
        self.pooling = tf.keras.layers.AveragePooling1D(pool_size, strides)

    def call(self, inputs, mask=None):

        n_batches, n_splits, n_tokens, hidden_size = inputs.shape
        new_shape = [-1, n_splits * n_tokens, hidden_size]
        flattened_inputs = tf.reshape(inputs, new_shape)
        if mask is not None:
            flattened_mask = tf.cast(
                tf.expand_dims(tf.reshape(mask, (-1, n_splits * n_tokens)), 2),
                tf.float32,
            )
            flattened_inputs = flattened_inputs * flattened_mask

        x = self.pooling(flattened_inputs)
        if mask is not None:
            pooled_mask = self.pooling(flattened_mask)
            x = tf.divide(x, pooled_mask + 1e-9)
        return x


class MaxMaskedPooling(tf.keras.layers.Layer):
    """Max pooling with strides and mask"""

    def __init__(self, config, *args, **kwargs):
        pool_size = config.pool_size
        strides = config.pool_strides
        num_labels = config.num_labels

        super().__init__(*args, **kwargs)
        self.pooling = tf.keras.layers.MaxPooling1D(pool_size, strides)

    def call(self, inputs, mask=None):
        n_batches, n_splits, n_tokens, hidden_size = inputs.shape
        new_shape = [-1, n_splits * n_tokens, hidden_size]
        flattened_inputs = tf.reshape(inputs, new_shape)
        if mask is not None:
            flattened_mask = (
                tf.cast(
                    tf.expand_dims(tf.reshape(mask, (-1, n_splits * n_tokens)), 2),
                    tf.float32,
                )
                - 1
            ) * 1e30
            flattened_inputs = flattened_inputs + flattened_mask

        x = self.pooling(flattened_inputs)

        return x

class AttentionPooling(tf.keras.layers.Layer):
    """Pooling using attention mechanism."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense_query = tf.keras.layers.Dense(50)
        self.attention = tf.keras.layers.Attention()

    def call(self, inputs, mask):
        n_batch, n_tokens, n_splits, n_dim = inputs.shape
        value = tf.reshape(inputs, [-1, n_tokens * n_splits, n_dim])
        x = tf.transpose(value, [0, 2, 1])
        query = tf.transpose(self.dense_query(x), [0, 2, 1])
        if mask is not None:
            mask = tf.reshape(mask, [-1, n_tokens * n_splits])
            mask =  [None, tf.cast(mask, bool)]
        output = self.attention([query, value], mask=mask)#, use_scale=True)
        return output

class PoolingClassificationHead(tf.keras.layers.Layer):
    """Classification head that pools BERT embeddings.
   
    The type of pooling can be 'mean', 'max' or an instance of
    a custom pooling mechanism subclassed from ``Layer``.

    This classification head should be combined with a model
    calculating cotextualised word embeddings such as ``FullTextBert``:

      >>> config = FullTextConfig(pool_type="max")
      >>> cls_token = 6
      >>> sep_token = 7
      >>> model = FullTextBert(config, cls_token=cls_token, sep_token=sep_token)
      >>> head = PoolingClassificationHead(config)
      >>> classifier = tf.keras.Sequential([model, head])
      >>> inputs = {
      ...     "input_ids": tf.constant([[1, 2, 3]]),
      ...     "attention_mask": tf.constant([[0, 0, 0]])}
      >>> output_logits = classifier(inputs)

    """

    def __init__(self, config, *args, **kwargs):
        """Create an instance of the classification head.

        Args:
            config: instance of FullTextConfig
        """
        dropout_rate = config.hidden_dropout_prob
        hidden_size = config.classification_hidden_size
        num_labels = config.num_labels
        pool_type = config.pool_type
        super().__init__(*args, **kwargs)
        if pool_type == "mean":
            self.pooling = MeanMaskedPooling(config)
        elif pool_type == "max":
            self.pooling = MaxMaskedPooling(config)
        elif issubclass(pool_type, tf.keras.layers.Layer):
            self.pooling = pool_type(config)
        self.flatten = tf.keras.layers.Flatten()
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation="relu")
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dense2 = tf.keras.layers.Dense(num_labels, activation="softmax")

    def call(self, inputs, mask):
        """Calculate the class logits from embeddings.

        Args:
            inputs: input embeddings (tf.Tensor)

        Returns:
            logits for each class (number of classes is configured by
            the config object)
        """
        x = self.pooling(inputs, mask)
        x = self.flatten(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x


class FullTextConfig(CamembertConfig):
    """Extra configuration for BERT for long texts."""

    model_type = "camembert"

    def __init__(
        self,
        pool_size=512,
        pool_strides=128,
        classification_hidden_size=50,
        pool_type="mean",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.classification_hidden_size = classification_hidden_size
        self.pool_type = pool_type
