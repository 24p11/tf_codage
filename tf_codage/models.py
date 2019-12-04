from transformers import TFBertForSequenceClassification, BertConfig
import tensorflow as tf

class BertForMultilabelClassification(TFBertForSequenceClassification):
    """Bert for sequence classification with sigmoid activation in the output layer"""
    
    def call(self, inputs, **kwargs):
        outputs = super().call(inputs, **kwargs)
        activation = tf.keras.layers.Activation('sigmoid')
        probs = activation(outputs[0])
        return probs