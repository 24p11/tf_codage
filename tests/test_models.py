from tf_codage.models import FullTextBert, TFCamembertModel
from tf_codage import models
from transformers import CamembertConfig
from transformers import CamembertTokenizer
import tensorflow as tf
import numpy as np
from numpy.testing import assert_allclose
from tempfile import mkdtemp

def test_full_text_bert_layer_id():
    """test selection of layer for classfication"""
    config = CamembertConfig(max_position_embeddings=514)
    model = FullTextBert(config, cls_token=5, sep_token=6, layer_id=-2)
    
    input_ids = tf.constant(
    [[3, 1, 0, 5, 6],
     [7, 2, 1, 1, 1],
     [0, 1, 0, 3, 3]])
    
    outputs, mask = model(input_ids)
    assert outputs.shape == (3, 10, 512, 768)
    assert mask is None
    
def test_full_text_bert():
    
    config = CamembertConfig(max_position_embeddings=514)
    
    model = FullTextBert(config, cls_token=5, sep_token=6)
    
    input_ids = tf.constant(
    [[3, 1, 0, 5, 6],
     [7, 2, 1, 1, 1],
     [0, 1, 0, 3, 3]])
    
    outputs, mask = model(input_ids)
    assert outputs.shape == (3, 10, 512, 768)
    assert mask is None
    
    # test with dict input
    outputs_dict, _ = model({'input_ids': input_ids})
    assert outputs_dict.shape == (3, 10, 512, 768)
    assert (outputs_dict.numpy() == outputs.numpy()).all()
    
    # test with attention mask
    outputs_attention, _ = model({'input_ids': input_ids,
                              'attention_mask': tf.ones((3, 5), tf.int32)})
    assert outputs.shape == (3, 10, 512, 768)
    # because of the padding
    assert not (outputs_attention.numpy() == outputs.numpy()).all()
    
def test_full_text_bert_attention_mask():
    
    config = CamembertConfig(max_position_embeddings=514)
    model = FullTextBert(config, cls_token=5, sep_token=6)
    
    input1 = tf.constant([[3, 4, 8, 7]])
    attention_mask1 = tf.constant([[1, 1, 1, 1]])
    input2 = tf.constant([[3, 4, 8, 7, 5]])
    attention_mask2a = tf.constant([[1, 1, 1, 1, 1]])
    attention_mask2b = tf.constant([[1, 1, 1, 1, 0]])
    
    
    out1, mask1 = model({'input_ids': input1,
                  'attention_mask': attention_mask1})
    out2a, mask2a = model({'input_ids': input2,
                  'attention_mask': attention_mask2a})
    out2b, mask2b = model({'input_ids': input2,
                  'attention_mask': attention_mask2b})
    
    
    # first 4 tokens should not be influenced by the last one if the mask set
    assert_allclose(out1.numpy()[0, 0, :5, :], out2b.numpy()[0, 0, :5, :])
    
    # last token should differ 
    assert not np.allclose(out1.numpy()[0, 0, 5, :], out2b.numpy()[0, 0, 5, :])
    
    # if the mask is not set, the last token my influence the rest
    assert not np.allclose(out1.numpy()[0, 0, :5, :], out2a.numpy()[0, 0, :5, :])
    
    # test masks
    expected_mask = np.zeros((1, 10, 512))
    expected_mask[0, 0, 1:5] = 1
    # special tokens are not masked
    expected_mask[:, :,  0] = 1
    expected_mask[:, :, -1] = 1
    assert np.equal(mask1.numpy(), expected_mask).all()
    assert np.equal(mask2b.numpy(), expected_mask).all()
    
    expected_mask[0, 0, 5] = 1
    assert_allclose(mask2a.numpy(), expected_mask)


def test_full_text_bert_compare():
    """Compare full text bert with batches of standard camembert"""
    
    model_dir = mkdtemp()
     
    config = CamembertConfig(max_position_embeddings=514)

    cls_token = 10
    sep_token = 11 
    max_batches = 2
    vocab_size = 10000
    
    # use random tokens as inputs
    multi_token = np.random.randint(1, vocab_size, size=(2, max_batches * 510)).tolist()
    
    def split_tokens(t):
        return [([cls_token] + t[i * 510:(i+1)*510] + [sep_token]) for i in range(max_batches)]
    
    bert_inputs = [split_tokens(tok_ids) for tok_ids in multi_token] 
    
    single_model = TFCamembertModel(config)
    out_bert = np.array([single_model(np.array(b))[0].numpy() for b in bert_inputs])

    single_model.save_pretrained(model_dir)
    fulltext_model = FullTextBert.from_pretrained(
        model_dir, cls_token=cls_token, sep_token=sep_token, max_batches=max_batches) 
    
    
    out_full_bert, _ = fulltext_model(np.array(multi_token))
    
    assert_allclose(out_bert, out_full_bert.numpy(), atol=1e-4)


def test_mean_masked_pooling_layer():
    """Test MeanMaskPoolingLayer"""
    
    n_batch = 4
    n_tokens = 16
    n_splits = 2
    n_hidden = 2
    hidden_inputs = np.random.randn(n_batch, n_tokens, n_splits, n_hidden).astype(np.float32)
    mask = np.ones((n_batch, n_tokens, n_splits))
    
    config = models.FullTextConfig(pool_size=32, pool_strides=1)
    pooling = models.MeanMaskedPooling(config)
    
    output = pooling(hidden_inputs, mask).numpy()
    expected = hidden_inputs.reshape(n_batch, n_tokens * n_splits, 1, n_hidden).mean(1)
    assert_allclose(expected, output)
    
    # test with different mask
    mask[:, 8:, :] = 0
    output = pooling(hidden_inputs, mask).numpy()
    expected = hidden_inputs[:, :8, :, :].reshape(n_batch, 8 * n_splits, 1, n_hidden).mean(1)
    assert_allclose(expected, output)
    
    # test non-overlapping strides
    config = models.FullTextConfig(pool_size=8, pool_strides=8)
    pooling = models.MeanMaskedPooling(config)
    mask = np.ones((n_batch, n_tokens, n_splits))
    
    output = pooling(hidden_inputs, mask).numpy()
    
    assert output.shape == (n_batch, 4, n_hidden)
    expected = hidden_inputs.reshape(n_batch, n_tokens * n_splits // 8, 8, n_hidden).mean(2)
    assert_allclose(expected, output)
    
    # test shape with overlapping strides
    config = models.FullTextConfig(pool_size=8, pool_strides=4)
    pooling = models.MeanMaskedPooling(config)
    
    output = pooling(hidden_inputs, mask).numpy()
    
    assert output.shape == (n_batch, 7, n_hidden)
    
def test_max_masked_pooling_layer():
    """Test MaxMaskPoolingLayer"""
    
    n_batch = 4
    n_tokens = 16
    n_splits = 2
    n_hidden = 2
    hidden_inputs = np.random.randn(n_batch, n_tokens, n_splits, n_hidden).astype(np.float32)
    mask = np.ones((n_batch, n_tokens, n_splits))
    
    config = models.FullTextConfig(pool_size=32, pool_strides=1)
    pooling = models.MaxMaskedPooling(config)
    
    output = pooling(hidden_inputs, mask).numpy()
    expected = hidden_inputs.reshape(n_batch, n_tokens * n_splits, 1, n_hidden).max(1)
    assert_allclose(expected, output)
    
    # test with different mask
    mask[:, 8:, :] = 0
    output = pooling(hidden_inputs, mask).numpy()
    expected = hidden_inputs[:, :8, :, :].reshape(n_batch, 8 * n_splits, 1, n_hidden).max(1)
    assert_allclose(expected, output)
