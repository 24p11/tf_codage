from transformers import TFBertForSequenceClassification, BertConfig
from transformers import CamembertTokenizer
from transformers import TFRobertaForSequenceClassification
from transformers import CamembertConfig
import tensorflow as tf

class TFCamembertForSequenceClassification(TFRobertaForSequenceClassification):
    config_class = CamembertConfig
    pretrained_model_archive_map = {}

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