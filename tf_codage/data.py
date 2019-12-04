from sklearn.preprocessing import  MultiLabelBinarizer
import tensorflow as tf
import pandas as pd
    
def load_into_dataframe(csv_path):
    """Load and preprocess CSV file"""

    df = pd.read_csv(csv_path, escapechar='\\', header=None)
    df.columns = ['encounter_num', 'acte', 'texte', 'instance_num']

    # shuffle the data frame
    df = df.sample(frac=1, random_state=13341).reset_index(drop=True)

    # transform into multi-labels
    gb = df.groupby(['encounter_num', 'texte'])
    df_new = gb['acte'].apply(set)
    df_new.name = 'target'
    df_encoded = df_new.reset_index()
    
    return df_encoded

def make_tokenize(tokenizer, max_len=512):
    """Make tokenize function that uses the tokenizer object"""
    def _tokenize(seq_1, seq_2=None):
        """Tokenize input sequeneces"""
        inputs = tokenizer.encode_plus(seq_1, seq_2, add_special_tokens=True, max_length=max_len)
        outputs = {}
        n_tokens = len(inputs['input_ids'])
        for k in ['input_ids', 'token_type_ids']:
            outputs[k] = inputs[k] + [0] * (max_len - n_tokens)
        outputs['attention_mask'] = [1] * len(inputs[k]) + [0] * (max_len - n_tokens)
        return outputs
    return _tokenize


def create_datasets_from_pandas(tokenize, phrase_1, target, batch_size=16, validation_split=0.1, phrase_2=None):
    """Create TF datasets (validation and training) from pandas series"""
    
    target_encoder = MultiLabelBinarizer().fit(target)
    tf_dtypes = ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
              tf.int32)
    tf_shapes = ({'input_ids': tf.TensorShape([None]),
                  'attention_mask': tf.TensorShape([None]),
                  'token_type_ids': tf.TensorShape([None])},
                 tf.TensorShape([None])
                )
    
    def _dataset_gen(data_slice):
        """make generator for data generation from a subset of data"""
        phrase_1_set = phrase_1[data_slice]
        if phrase_2:
            phrase_2_set = phrase_2[data_slice]
        target_set = target[data_slice]
        
        def _generator():
            if not phrase_2:
                tokens = (tokenize(r) for r in phrase_1_set)
            else:
                tokens = (tokenize(r_1, r_2) 
                          for r1, r2 in zip(phrase_1_set, phrase_2_set))
            return zip(tokens, 
                       target_encoder.transform(target_set.to_list()))
        return _generator
    
    n_train = int(len(target) * (1 - validation_split))
    
    train_set_gen = _dataset_gen(slice(None, n_train))
    validation_set_gen = _dataset_gen(slice(n_train, None))
    
    train_set = tf.data.Dataset.from_generator(
        train_set_gen, tf_dtypes, tf_shapes)
    validation_set = tf.data.Dataset.from_generator(
        validation_set_gen, tf_dtypes, tf_shapes)
    
    train_set = train_set.cache().batch(batch_size)
    validation_set = validation_set.cache().batch(batch_size)

    return train_set, validation_set