import tensorflow as tf


class WeightedBinaryCrossEntropyMax(tf.keras.losses.Loss):
    """
    Args:
      first_token_id: Scalar to affect the positive labels of the loss function.
      unk_token_id: Scalar to affect the entirety of the loss function.
      from_logits: Whether to compute loss form logits or the probability.
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(self, first_token_id, unk_token_id, reduction=tf.keras.losses.Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super(WeightedBinaryCrossEntropy, self).__init__(reduction=reduction,
                                                         name=name)
        self.first_token_id = tf.constant([first_token_id], dtype=tf.int64)
        self.unk_token_id = tf.constant([unk_token_id], dtype=tf.int64)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction='none')

    def call(self, y_true, y_pred):
            
        
        #mask = tf.math.logical_not(tf.math.equal(y_true, self.first_token_id))
        #mask = tf.math.greater(y_true, self.first_token_id) 
        mask =  tf.logical_or( tf.math.greater(y_true, self.first_token_id) ,tf.equal(y_true, self.unk_token_id) )
        
        loss_ = self.loss_object(y_true, y_pred)
    
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
    
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    """
    Args:
      first_token_id: Scalar to affect the positive labels of the loss function.
      unk_token_id: Scalar to affect the entirety of the loss function.
      from_logits: Whether to compute loss form logits or the probability.
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(self, first_token_id, reduction=tf.keras.losses.Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super(WeightedBinaryCrossEntropy, self).__init__(reduction=reduction,
                                                         name=name)
        
        self.first_token_id = tf.constant([first_token_id], dtype=tf.int64)
        
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction='none')

    def call(self, y_true, y_pred):
            
        
        mask = tf.math.logical_not(tf.math.equal(y_true, self.first_token_id))
        
        loss_ = self.loss_object(y_true, y_pred)
    
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
    
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)