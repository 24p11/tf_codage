import os
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, OneHotEncoder
import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import re
import csv
from itertools import islice


def load_into_dataframe(csv_path, shuffle=True, **kwargs):
    """Load and preprocess CSV file"""

    df = pd.read_csv(csv_path, escapechar="\\", header=None, **kwargs)
    if len(df.columns) == 4:
        column_names = ["encounter_num", "acte", "texte", "instance_num"]
    elif len(df.columns) == 5:
        column_names = ["encounter_num", "acte", "texte", "instance_num", "libelle"]

    df.columns = column_names

    # shuffle the data frame
    if shuffle:
        df = df.sample(frac=1, random_state=13341).reset_index(drop=True)
    return df


def dataframe_to_multilabel(df):
    """Add multilabel target to data frame"""

    # transform into multi-labels
    gb = df.groupby(["encounter_num", "texte"])
    df_new = gb["acte"].apply(set)
    df_new.name = "target"
    df_encoded = df_new.reset_index()

    return df_encoded


def dataframe_to_paired_sentences(df, fraction_data=1.0):
    """Add columns for paired sentences task"""

    df = df[~df.libelle.isnull()]

    # keep only top codes
    top_acte = df.acte.value_counts().index[:10]
    # df = df[df.acte.isin(top_acte)]

    df = df.reset_index(drop=True)
    df["target"] = 1  # for matching sentences

    # add shuffled definitions
    shuffled_libelle = (
        df.libelle.sample(frac=1, random_state=13341)
        .reset_index()
        .drop("index", axis=1)
    )

    df2 = df.copy()
    df2.libelle = shuffled_libelle

    df2["target"] = 0  # for not matching

    df_union = pd.concat([df, df2], axis=0).reset_index(drop=True)

    # randomly shuffle rows
    df_union = df_union.sample(frac=fraction_data, random_state=93313).reset_index(
        drop=True
    )

    return df_union


def aggregate_classes(df, column="acte", other_name="OTHER", min_examples=30):
    """Combine infrequent classes into a common class"""

    column = "acte"
    examples_per_label = df[column].value_counts()
    small_labels = examples_per_label[
        examples_per_label < min_examples
    ].index.to_numpy()
    df = df.copy()

    df.loc[df[column].isin(small_labels), column] = "OTHER"

    return df


def make_multilabel_dataframe(csv_path, min_examples=30, use_classes=None):
    """Make multilabel dataframe.
    
    use_classes: which classes to return, the classes which are not in the
    list will be replace with 'OTHER'"""

    if isinstance(csv_path, pd.DataFrame):
        df = csv_path
    else:
        df = load_into_dataframe(csv_path, shuffle=False)

    if use_classes is not None:
        df.loc[~df.acte.isin(use_classes), "acte"] = "OTHER"
    df = aggregate_classes(df, min_examples=min_examples)
    num_labels = int(df.acte.value_counts().count())
    df = dataframe_to_multilabel(df)
    df = df.sample(frac=1, random_state=13341).reset_index(drop=True)

    return df, num_labels


def make_paired_sentences_dataframe(csv_path, fraction_data=1.0):
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
            seq_1,
            seq_2,
            add_special_tokens=True,
            max_length=max_len,
            truncation_strategy="only_first",
        )
        outputs = {}
        n_tokens = len(inputs["input_ids"])
        for k in ["input_ids", "token_type_ids"]:
            outputs[k] = inputs[k] + [0] * (max_len - n_tokens)
        outputs["attention_mask"] = [1] * len(inputs[k]) + [0] * (max_len - n_tokens)
        return outputs

    return _tokenize


class DummyEncoder:
    def transform(self, input):
        return input


def create_datasets_from_pandas(
    tokenize,
    phrase_1,
    target,
    batch_size=16,
    validation_split=0.1,
    phrase_2=None,
    encoder="multilabel_binarizer",
):
    """Create TF datasets (validation and training) from pandas series"""

    if encoder == "multilabel_binarizer":
        target_encoder = MultiLabelBinarizer().fit(target)
    elif encoder == "label_binarizer":
        target_encoder = LabelBinarizer().fit(target)
    elif hasattr(encoder, "transform"):
        target_encoder = encoder
    else:
        target_encoder = DummyEncoder()

    encoded_value = target_encoder.transform([target[0]])
    ndim = np.asarray(encoded_value).ndim

    if ndim == 2:
        shape = [None]
    else:
        shape = []

    tf_dtypes = (
        {"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32},
        tf.int32,
    )
    tf_shapes = (
        {
            "input_ids": tf.TensorShape([None]),
            "attention_mask": tf.TensorShape([None]),
            "token_type_ids": tf.TensorShape([None]),
        },
        tf.TensorShape(shape),
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
                tokens = (
                    tokenize(r_1, r_2) for r_1, r_2 in zip(phrase_1_set, phrase_2_set)
                )
            return zip(tokens, target_encoder.transform(target_set.to_list()))

        return _generator

    n_train = int(len(target) * (1 - validation_split))

    train_set_gen = _dataset_gen(slice(None, n_train))
    validation_set_gen = _dataset_gen(slice(n_train, None))

    train_set = tf.data.Dataset.from_generator(train_set_gen, tf_dtypes, tf_shapes)
    validation_set = tf.data.Dataset.from_generator(
        validation_set_gen, tf_dtypes, tf_shapes
    )

    train_set = train_set.cache().batch(batch_size)
    validation_set = validation_set.cache().batch(batch_size)

    return train_set, validation_set, target_encoder


def create_sentence_dataset(
    documents,
    sentence_tokenizer,
    word_tokenizer,
    sentence_embedding_model,
    max_sentences=512,
):
    def generator():

        for doc in documents:
            sentences = sentence_tokenizer.tokenize(doc)
            sentences = sentences[:max_sentences]
            sentences = [s.lower() for s in sentences]

            tokens = word_tokenizer.batch_encode_plus(
                sentences,
                add_special_tokens=True,
                max_length=sentence_embedding_model.config.max_position_embeddings - 2,
                pad_to_max_length=False,
                return_tensors="tf",
                return_attention_masks=True,
            )
            del tokens["token_type_ids"]
            sentence_embedding = sentence_embedding_model(tokens)
            sentence_mask = tf.ones((sentence_embedding.shape[0],), dtype=tf.int32)
            yield {
                "inputs_embeds": sentence_embedding,
                "attention_mask": sentence_mask,
            }

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types={"inputs_embeds": tf.float32, "attention_mask": tf.int32},
        output_shapes={
            "inputs_embeds": tf.TensorShape([None, None]),
            "attention_mask": tf.TensorShape([None]),
        },
    )

    return dataset


def list_cmd_codes(filename_template, files_dir=".", cmd_fmt="([0-9]+)"):
    """Find a list of cmd codes parsed from filenames.
    
    Examples:
    
    >>> list_cmd_codes('dummy-actes-cmd-{}.csv', files_dir='tests/data')
    ['02', '03', '06', '12']
    
    >>> list_cmd_codes('no-actes-cmd-{}.csv', files_dir='tests/data')
    []
    
    You can also specify the format for the CMD code. For non-numeric format,
    
    >>> list_cmd_codes('dummy-actes-cmd-{}.csv', files_dir='tests/data', cmd_fmt='([a-z]+)')
    ['sample']
    
    """
    actes = glob.glob(os.path.join(files_dir, filename_template.format("*")))

    def match_cmd(p):
        m = re.match(filename_template.format(cmd_fmt), os.path.split(p)[1])
        if m:
            return m.groups(0)[0]
        else:
            return None

    CMDs = [match_cmd(p) for p in actes]

    # filter no-matches and sort

    CMDs = sorted([cmd for cmd in CMDs if cmd])

    return CMDs


class CSVDataReader:
    """Read data from a CSV file. 
    
    The class reads data row-by-row and it does not load the whole file into memory."""

    def __init__(
        self,
        csv_path,
        columns,
        skip_rows=0,
        max_rows=None,
        doublequote=False,
        delimeter=",",
    ):

        self.csv_path = csv_path
        self.columns = columns
        self.skip_rows = skip_rows
        self.max_rows = max_rows
        self.doublequote = doublequote
        self.delimeter = delimeter

    # we are using iterable class rather than generator so that the
    # iteration can be restarted (for example, to train over multiple
    # epochs)
    def __iter__(self):

        delimeter = self.delimeter
        csv_path = self.csv_path
        columns = self.columns
        skip_rows = self.skip_rows
        max_rows = self.max_rows
        doublequote = self.doublequote

        def wrapper(gen):
            while True:
                try:
                    yield next(gen)
                except StopIteration:
                    break
                except csv.Error as e:
                    print(
                        "skipping line {} due to an error: {}".format(gen.line_num, e),
                        file=sys.stderr,
                    )

        n_columns = None

        with open(csv_path, newline="") as f:

            # use CSV reader from standard library with custom dialect
            # https://docs.python.org/3/library/csv.html#dialects-and-formatting-parameters
            reader = csv.reader(
                f, doublequote=doublequote, escapechar="\\", quoting=csv.QUOTE_ALL
            )

            # get the header and parse column names
            header = next(reader)
            column_names = header

            columns_idx = [
                column_names.index(col) if isinstance(col, str) else col
                for col in columns
            ]
            
            for i in range(skip_rows):
                f.readline()

            for row in islice(wrapper(reader), 0, max_rows):
                if n_columns is None:
                    n_columns = len(row)
                else:
                    assert n_columns == len(
                        row
                    ), "wrong number of columns at line {}".format(reader.line_num)
                yield [row[i] for i in columns_idx]


def make_transformers_dataset(
    examples, transform_inputs=None, transform_labels=None, batch_size=None
):
    """make tf dataset compatible with transformers models.
    
    `examples` : iterable of tuples
        an iterable generating training (or validation) examples
    `transform_inputs` : function
        a function to transform a set of columns from examples into a dict 
        with keys `input_ids` (token indices), `attention_mask` and
        `token_type_ids`
    `transform_labels` : function
        a function that transforms a set of columns from examples
        into one-hot representation of the label
    `batch_size` : integer
        size of a batch
    """

    if transform_inputs is None:
        transform_inputs = lambda X: X
    if transform_labels is None:
        transform_labels = lambda y: y

    def data_generator():

        for training_example in examples:
            token_data = transform_inputs(training_example)
            label_enc = transform_labels(training_example)

            yield token_data, label_enc

    sample_tokens, sample_labels = next(data_generator())
    num_labels = len(sample_labels)
    n_tokens = len(sample_tokens["input_ids"])

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_types=(
            {
                "input_ids": tf.int32,
                "attention_mask": tf.int32,
                "token_type_ids": tf.int32,
            },
            tf.int32,
        ),
        output_shapes=(
            {
                "input_ids": tf.TensorShape([n_tokens]),
                "attention_mask": tf.TensorShape([n_tokens]),
                "token_type_ids": tf.TensorShape([n_tokens]),
            },
            tf.TensorShape([num_labels]),
        ),
    )
    if batch_size:
        dataset = dataset.batch(batch_size).prefetch(2)

    return dataset
