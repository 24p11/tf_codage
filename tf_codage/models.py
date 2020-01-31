import os
import GPUtil
from transformers import TFBertForSequenceClassification, BertConfig
from transformers import CamembertTokenizer
from transformers import TFRobertaForSequenceClassification
from transformers import TFRobertaModel
from transformers import CamembertConfig
import tensorflow as tf

os.environ['CUDA_DEVICE_ORDER'] = os.environ.get('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
try:
    available_devices = str(GPUtil.getFirstAvailable(maxMemory=0.1)[0])
except:
    available_devices = ''
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', available_devices)

class TFCamembertForSequenceClassification(TFRobertaForSequenceClassification):
    config_class = CamembertConfig
    pretrained_model_archive_map = {}
    
class TFCamembertModel(TFRobertaModel):
    config_class = CamembertConfig
    pretrained_model_archive_map = {}
    
class FullTextBert(TFCamembertModel):
    
    def __init__(self, config, *inputs, cls_token=None, sep_token=None, max_batches=10, layer_id=-1, **kwargs):
        """
        
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
    
    def call(self, inputs, **kwargs):
        cls_token = self.cls_token
        sep_token = self.sep_token
        
        max_batches = self.max_batches
        input_size = 512
        hidden_size = self.config.hidden_size
        
        # take account of special tokens
        bert_seq_len = input_size - 2
        
        if isinstance(inputs, dict):
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask', None)
        else:
            input_ids = inputs
            attention_mask = None
        
        
        n_seq, seq_len = input_ids.shape
        padded_inputs = tf.pad(input_ids, [[0, 0], [0, max_batches * bert_seq_len - seq_len]], constant_values=1)
        
        new_tensor = tf.reshape(padded_inputs, [-1, bert_seq_len])
        
        padded_tensor = tf.pad(new_tensor, [[0, 0], [1,0]], constant_values=cls_token)
        padded_tensor = tf.pad(padded_tensor, [[0, 0], [0,1]], constant_values=sep_token)
        
        if attention_mask is not None:
            padded_mask = tf.pad(attention_mask, [[0, 0], [0, max_batches * bert_seq_len - seq_len]], constant_values=0)
            attention_mask = tf.reshape(padded_mask, [-1, bert_seq_len])
            attention_mask = tf.pad(attention_mask, [[0, 0], [1, 1]], constant_values=1)
            
        
        outputs = self.roberta(padded_tensor,
                               attention_mask=attention_mask,
                               **kwargs)
        
        if self.layer_id == -1:
            sequence_output = outputs[0]
        else:
            sequence_output = outputs[2][self.layer_id]
        
        reshaped_outputs = tf.reshape(sequence_output, [-1, max_batches, input_size, hidden_size])
        
        if attention_mask is not None:
            reshaped_mask = tf.reshape(attention_mask, [-1, max_batches, input_size])
        else:
            reshaped_mask = None
            
        return reshaped_outputs, reshaped_mask

class BertForMultilabelClassification(TFBertForSequenceClassification):
    """Bert for sequence classification with sigmoid activation in the output layer"""
    
    def call(self, inputs, **kwargs):
        outputs = super().call(inputs, **kwargs)
        activation = tf.keras.layers.Activation('sigmoid')
        probs = activation(outputs[0])
        return probs
    
class CamembertForMultilabelClassification(TFCamembertForSequenceClassification):
    """Camembert for sequence classification with sigmoid activation in the output layer"""
    
    def call(self, inputs, **kwargs):
        outputs = super().call(inputs, **kwargs)
        activation = tf.keras.layers.Activation('sigmoid')
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
                tf.expand_dims(
                    tf.reshape(mask, (-1, n_splits * n_tokens)),
                    2),
                tf.float32)
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
            flattened_mask = (tf.cast(
                tf.expand_dims(
                    tf.reshape(mask, (-1, n_splits * n_tokens)),
                    2),
                tf.float32) - 1) * 1e30
            flattened_inputs = flattened_inputs + flattened_mask
            
        x = self.pooling(flattened_inputs)
        
        return x
    
class PoolingClassificationHead(tf.keras.layers.Layer):
    def __init__(self, config, *args, **kwargs):
        dropout_rate = config.hidden_dropout_prob
        hidden_size = config.classification_hidden_size
        num_labels = config.num_labels
        pool_type = config.pool_type
        super().__init__(*args, **kwargs)
        if pool_type == 'mean':
            self.pooling = MeanMaskedPooling(config)
        elif pool_type == 'max':
            self.pooling = MaxMaskedPooling(config)
        self.flatten = tf.keras.layers.Flatten()
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dense2 = tf.keras.layers.Dense(num_labels, activation='softmax')
        
    
    def call(self, inputs, mask):
        x = self.pooling(inputs)
        x = self.flatten(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x
    
class FullTextConfig(CamembertConfig):
    
    model_type = 'camembert'
    
    def __init__(self, pool_size=512, pool_strides=128, classification_hidden_size=50, pool_type='mean', **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.classification_hidden_size = classification_hidden_size
        self.pool_type = pool_type
        