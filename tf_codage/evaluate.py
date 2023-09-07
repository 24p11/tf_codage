import numpy as np
import tensorflow as tf

def print_support(encoded):
    """Print support given binarised labels"""

    encoded = np.asarray(encoded)
    support = encoded.sum(0)

    print("    n° classes")
    print("")
    print("all {:>10d}".format(encoded.shape[1]))
    print("> 0 ex {:>7d}".format((support > 0).sum()))
    print("> 10 ex {:>6d}".format((support > 10).sum()))
    print("")
    print("       support")
    print(" (n° examples)")
    print("")
    print("sum {:>10d}".format(np.sum(support)))
    print("median {:>7.1f}".format(np.median(support)))
    print("max {:>10d}".format(np.max(support)))
    print("min {:>10d}".format(np.min(support)))


def get_encoded_array(dataset):
    """Take TF dataset and return an array of binarised labels"""
    real_encoded = np.vstack([s[1].numpy() for s in dataset])

    return real_encoded

def isin(padded_outputs,padded_labels):
    """Percentage of time logits contains labels on non-0s."""
    with tf.name_scope("isin"):
        tile_multiple = tf.shape(padded_labels)[-1]
        tiled_outputs = tf.tile(tf.expand_dims(padded_outputs, axis=-1), multiples=[1, 1, tile_multiple])
        tiled_labels = tf.reshape(tf.tile(padded_labels, multiples=[1, tile_multiple]), [-1, tile_multiple, tile_multiple])
        equal = tf.equal(tiled_outputs, tiled_labels)
        any = tf.reduce_any(equal, axis=-1)
        return any
    
def _pad_tensors_to_same_length(x, y):
    """Pad x and y so that the results have the same length (second dimension)."""
    x_length = tf.shape(x)[1]
    y_length = tf.shape(y)[1]

    max_length = tf.maximum(x_length, y_length)

    x = tf.pad(x, [[0, 0], [0, max_length - x_length]])
    y = tf.pad(y, [[0, 0], [0, max_length - y_length]])
    return x, y, max_length.numpy()

def seq2seq_true_false_positives(pred, labels, first_token_id = 3, unk_token_id = 1):
    
    pred, labels, max_length = _pad_tensors_to_same_length(pred, labels)
    padded_pred = tf.cast(pred, tf.int32)
    padded_labels = tf.cast(labels, tf.int32)
    
    first_token_id_ = tf.constant([first_token_id])
    unk_token_id_ =  tf.constant([unk_token_id])
    
    #weights_pred = tf.logical_and(tf.not_equal(padded_pred, 0), tf.not_equal(padded_pred, 1))
    #weights_labels = tf.logical_and(tf.not_equal(padded_labels, 0), tf.not_equal(padded_labels, 1))

    weights_pred =  tf.logical_or( tf.math.greater(padded_pred, first_token_id_) ,tf.equal(padded_pred, unk_token_id_) )
    weights_labels = tf.logical_or( tf.math.greater(padded_labels, first_token_id) ,tf.equal(padded_labels, unk_token_id_) ) 

    out_in_lab = tf.boolean_mask(isin(padded_pred, padded_labels ), weights_pred)
    lab_in_out = tf.boolean_mask(isin(padded_labels, padded_pred ), weights_labels)

    true_positives = tf.cast(out_in_lab, tf.float32)
    false_positives = tf.cast(tf.logical_not(out_in_lab), tf.float32)
    false_negatives = tf.cast(tf.logical_not(lab_in_out), tf.float32)

    return tf.math.reduce_sum(true_positives), tf.math.reduce_sum(false_positives), tf.math.reduce_sum(false_negatives)

def seq2seq_accuracy(pred, labels):
    
    pred_pad, labels_pad, max_length = _pad_tensors_to_same_length(pred, labels)
    
    acc1 = tf.math.equal(pred_pad,labels_pad)
    acc2 = tf.equal(tf.math.reduce_sum(tf.cast(acc1, tf.float32),axis =-1),max_length)
    acc= tf.math.reduce_sum(tf.cast(acc2, tf.float32))
    
    return acc