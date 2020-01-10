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
    
class FullTextBert(TFCamembertModel):
    
    def __init__(self, config, *inputs, cls_token=None, sep_token=None, max_batches=10, **kwargs):
        
        super().__init__(config, *inputs, **kwargs)
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.max_batches = max_batches
    
    def call(self, input_ids, **kwargs):
        cls_token = self.cls_token
        sep_token = self.sep_token
        
        max_batches = self.max_batches
        input_size = 512
        
        # take account of special tokens
        bert_seq_len = input_size - 2
        n_seq, seq_len = input_ids.shape
        
        padded_inputs = tf.pad(input_ids, [[0, 0], [0, max_batches * bert_seq_len - seq_len]], constant_values=1)
        
        new_tensor = tf.reshape(padded_inputs, [n_seq * max_batches, bert_seq_len])
        
        padded_tensor = tf.pad(new_tensor, [[0, 0], [1,0]], constant_values=cls_token)
        padded_tensor = tf.pad(padded_tensor, [[0, 0], [0,1]], constant_values=sep_token)
        
        outputs = self.roberta(padded_tensor, **kwargs)
        
        reshaped_outputs = tf.reshape(outputs[0], [n_seq, max_batches, input_size, -1])
        return reshaped_outputs

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