import os
from sklearn.preprocessing import  MultiLabelBinarizer, LabelBinarizer, OneHotEncoder
import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import re
    
def load_into_dataframe(csv_path, shuffle=True, **kwargs):
    """Load and preprocess CSV file"""

    df = pd.read_csv(csv_path, escapechar='\\', header=None, **kwargs)
    if len(df.columns) == 4:
        column_names = ['encounter_num', 'acte', 'texte', 'instance_num']
    elif len(df.columns) == 5:
        column_names = ['encounter_num', 'acte', 'texte', 'instance_num', 'libelle']
        
    df.columns = column_names

    # shuffle the data frame
    if shuffle:
        df = df.sample(frac=1, random_state=13341).reset_index(drop=True)
    return df

def dataframe_to_multilabel(df):
    """Add multilabel target to data frame"""
    
    # transform into multi-labels
    gb = df.groupby(['encounter_num', 'texte'])
    df_new = gb['acte'].apply(set)
    df_new.name = 'target'
    df_encoded = df_new.reset_index()
    
    return df_encoded

def dataframe_to_paired_sentences(df, fraction_data=1.):
    """Add columns for paired sentences task"""
    
    df = df[~df.libelle.isnull()]

    # keep only top codes
    top_acte = df.acte.value_counts().index[:10]
    #df = df[df.acte.isin(top_acte)]

    df = df.reset_index(drop=True)
    df['target'] = 1 # for matching sentences
    
    # add shuffled definitions
    shuffled_libelle = df.libelle.sample(frac=1, random_state=13341).reset_index().drop('index', axis=1)

    df2 = df.copy()
    df2.libelle = shuffled_libelle

    df2['target'] = 0 # for not matching

    df_union = pd.concat([df, df2], axis=0).reset_index(drop=True)

    # randomly shuffle rows
    df_union = df_union.sample(frac=fraction_data, random_state=93313).reset_index(drop=True)
    
    return df_union

def aggregate_classes(df, column='acte', other_name='OTHER', min_examples=30):
    """Combine infrequent classes into a common class"""
    
    column = 'acte'
    examples_per_label = df[column].value_counts()
    small_labels = examples_per_label[examples_per_label < min_examples].index.to_numpy()
    df = df.copy()

    df.loc[df[column].isin(small_labels), column] = 'OTHER'
    
    return df
    
def make_multilabel_dataframe(csv_path, min_examples=30, use_classes=None):
    """Make multilabel dataframe.
    
    use_classes: which classes to return, the classes which are not in the
    list will be replace with 'OTHER'"""
    
    df = load_into_dataframe(csv_path, shuffle=False)
    if use_classes is not None:
        df.loc[~df.acte.isin(use_classes), 'acte'] = 'OTHER'
    df = aggregate_classes(df, min_examples=min_examples)
    num_labels = int(df.acte.value_counts().count())
    df = dataframe_to_multilabel(df)
    df = df.sample(frac=1, random_state=13341).reset_index(drop=True)
    
    return df, num_labels

def make_paired_sentences_dataframe(csv_path, fraction_data=1.):
    """Make paired sentences dataframe"""
    
    df = load_into_dataframe(csv_path)
    df = dataframe_to_paired_sentences(df, fraction_data=fraction_data)
    num_labels = int(df.target.value_counts().count())
    
    return df, num_labels

def make_tokenize(tokenizer, max_len=512):
    """Make tokenize function that uses the tokenizer object"""
    def _tokenize(seq_1, seq_2=None):
        """Tokenize input sequeneces"""
        inputs = tokenizer.encode_plus(
            seq_1, seq_2, 
            add_special_tokens=True, max_length=max_len,
            truncation_strategy='only_first')
        outputs = {}
        n_tokens = len(inputs['input_ids'])
        for k in ['input_ids', 'token_type_ids']:
            outputs[k] = inputs[k] + [0] * (max_len - n_tokens)
        outputs['attention_mask'] = [1] * len(inputs[k]) + [0] * (max_len - n_tokens)
        return outputs
    return _tokenize

class DummyEncoder:
    
    def transform(self, input):
        return input
    
def create_datasets_from_pandas(tokenize, phrase_1, target, batch_size=16, validation_split=0.1, phrase_2=None, encoder='multilabel_binarizer'):
    """Create TF datasets (validation and training) from pandas series"""
    
    if encoder == 'multilabel_binarizer':
        target_encoder = MultiLabelBinarizer().fit(target)
    elif encoder == 'label_binarizer':
        target_encoder = LabelBinarizer().fit(target)
    elif hasattr(encoder, 'transform'):
        target_encoder = encoder
    else:
        target_encoder = DummyEncoder()
        
    encoded_value = target_encoder.transform([target[0]])
    ndim = np.asarray(encoded_value).ndim
    
    if ndim == 2:
        shape = [None]
    else:
        shape = []
        
    tf_dtypes = ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
              tf.int32)
    tf_shapes = ({'input_ids': tf.TensorShape([None]),
                  'attention_mask': tf.TensorShape([None]),
                  'token_type_ids': tf.TensorShape([None])},
                 tf.TensorShape(shape)
                )
    
    def _dataset_gen(data_slice):
        """make generator for data generation from a subset of data"""
        phrase_1_set = phrase_1[data_slice]
        
        if phrase_2 is not None:
            phrase_2_set = phrase_2[data_slice]
        target_set = target[data_slice]
        
        def _generator():
            if phrase_2 is None:
                tokens = (tokenize(r) for r in phrase_1_set)
            else:
                tokens = (tokenize(r_1, r_2) 
                          for r_1, r_2 in zip(phrase_1_set, phrase_2_set))
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

    return train_set, validation_set, target_encoder


def list_cmd_codes(filename_template, files_dir='.'):
    """Find a list of cmd codes parsed from filenames.
    
    Example:
    
    >>> list_cmd_codes('dummy-actes-cmd-{}.csv', files_dir='tests/data')
    ['02', '03', '06', '12']
    """
    actes = glob.glob(os.path.join(files_dir, filename_template.format('*')))

    def match_cmd(p):
        m = re.match(filename_template.format('([0-9]+)'), os.path.split(p)[1])
        if m:
            return m.groups(0)[0]
        else:
            return None

    CMDs = sorted([match_cmd(p) for p in actes])
    
    return CMDs