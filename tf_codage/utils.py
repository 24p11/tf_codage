"""Set of helper functions to work with files, standard output etc. They are not specific to machine learning or medical informatics.
"""

import sys
from contextlib import redirect_stdout
import subprocess
import re
import pandas as pd
import os
from itertools import chain, islice
from pathlib import Path
import pickle
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf

class TeeStream:
    """Redirect stream to many different outputs.
    
    This is required so that we can see the progress bar 
    both in the notebook and in the terminal (for runs
    from command line)"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def notebook_copy_stdout():
    """Context manager to copy the notebook output to standard output (terminal)"""
    tee = TeeStream(sys.stdout, sys.__stdout__)
    redirect = redirect_stdout(tee)
    return redirect


def print_console(*args, **kwargs):
    """Print message to console and notebook output.

    It can be used as a drop-in replacement for the standard
    print function:
        
        >>> from tf_codage.utils import print_console as print
     
    """

    print(*args, **kwargs, file=sys.stdout)
    print(*args, **kwargs, file=sys.__stdout__)


def download_hdfs(input_file, output_file):
    """Download files from HDFS store."""
    import sys
    import pyarrow

    conn = pyarrow.hdfs.connect()

    output_file = sys.argv[1]
    input_file = sys.argv[1]

    filenames = conn.ls(input_file)
    filenames = [f for f in filenames if f.endswith(".csv")]
    print(filenames)

    output_file = open(output_file, "wb")
    for f in filenames:
        f = f.replace("hdfs://bbsedsi", "")
        print("Downloading {}".format(f))
        fid = conn.open(f)
        output_file.write(fid.read())
    output_file.close()


def save_model(
    model,
    model_name,
    metric_data=None,
    tokenizer=None,
    encoder=None,
    root_dir="..",
    formats=("transformers",),
):
    """Save model to subdirs of `root_dir` together with the metrics, encoder and tokenzier if specified."""

    import os

    model_dir = Path(root_dir, "models", model_name)
    print("Saving model files to ", model_dir)

    if "transformers" in formats:
        transformers_path = model_dir / "pretrained_model_transformers"
        os.makedirs(transformers_path, exist_ok=True)
        model.save_pretrained(transformers_path)

    if "tf" in formats:
        model.save(model_dir / "trained_model.tf")

    if metric_data:

        metrics_dir = os.path.join(root_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        metric_dict = dict(zip(model.metrics_names, map(float, metric_data)))

        import json

        with open(os.path.join(metrics_dir, model_name + ".json"), "w") as fid:
            json.dump(metric_dict, fid)

    if tokenizer:
        tokenizer.save_pretrained(model_dir)

    if encoder:
        import joblib

        joblib.dump(encoder, os.path.join(model_dir, "encoder.joblib"))


def grep_keras_results_from_notebook(ipynb_file, filter_by="val_loss"):
    """Parse keras results from jupyter notebook.
    
    Example:

        >>> df = grep_keras_results_from_notebook(
        ...     'tests/data/dummy_notebook.ipynb')
        >>> print(df)
                 loss accuracy val_loss val_accuracy
        epoch                                       
        1      0.0000   0.4889   0.0000       0.4000
        2      0.0000   0.4889   0.0000       0.4000
    """
    pr = subprocess.run(["grep", filter_by, ipynb_file], capture_output=True)
    out = pr.stdout

    lines = out.decode("utf-8").splitlines()

    rows = []
    i = 0
    for line in lines:
        m = re.findall(r"([a-z0-9\._]*): ([0-9]+\.[0-9]*)", line)
        if m:
            i += 1
            m += [("epoch", i)]
            rows.append(dict(m))
    df = pd.DataFrame(rows).set_index("epoch")
    return df


def batch_generator(l, batch_size):
    """Split list l into batches of size batch_size. 
    
    It returns a generator of generators.
    
    Example:
        >>> points = [1, 2, 3, 4, 5]
        >>> [list(batch) for batch in batch_generator(points, 2)]
        [[1, 2], [3, 4], [5]]
    """

    l_iter = iter(l)

    def single_batch():
        yield next_item
        try:
            for i in range(batch_size - 1):
                yield next(l_iter)
        except StopIteration:
            return

    while True:

        try:
            next_item = next(l_iter)
            yield single_batch()
        except StopIteration:
            break


def split_file(input_file, out_dir, lines_per_split):
    """Split input_file into multiple files writen in out_dir each with lines_per_split lines.
    
    Save header at the top of each file."""

    def chunks(iterable, n):
        "chunks(ABCDE,2) => AB CD E"
        iterable = iter(iterable)
        while True:
            try:
                yield chain([next(iterable)], islice(iterable, n - 1))
            except StopIteration:
                return

    _, filename = os.path.split(input_file)

    os.makedirs(out_dir, exist_ok=True)

    split_filenames = []
    with open(input_file) as fid:
        header = fid.readline()

        for i, lines in enumerate(chunks(fid, lines_per_split)):
            file_split = os.path.join(
                out_dir, "{}.split-{:04d}.csv".format(filename, i)
            )
            split_filenames.append(file_split)
            with open(file_split, "w") as f:
                f.write(header)
                f.writelines(lines)
    return split_filenames

def save_tokenizer(tokenizer,PATH):
    pickle.dump({'config': tokenizer.get_config(),
             'weights': tokenizer.get_weights()}
            , open(PATH, "wb"))
    

def load_tokenizer(PATH):
    from_disk = pickle.load(open(PATH, "rb"))
    tok = TextVectorization.from_config(from_disk['config'])
    # You have to call `adapt` with some dummy data (BUG in Keras)
    tok.adapt( tf.data.Dataset.from_tensor_slices(["xyz"]))
    tok.set_weights(from_disk['weights'])
    return tok



###Tensor operations

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq

def set_tensor_by_indices_to_value(input_tensor, indices, value):
    """
    from : https://github.com/tensorflow/models/blob/master/official/nlp/modeling/ops/sampling_module.py
    Where indices is True, set the value in input_tensor to value.

    Args:
    input_tensor: float (batch_size, dim)
    indices: bool (batch_size, dim)
    value: float scalar

    Returns:
    output_tensor: same shape as input_tensor.
    """
    value_tensor = tf.zeros_like(input_tensor) + value
    output_tensor = tf.where(indices, value_tensor, input_tensor)
    
    return output_tensor

def pad_tensors_to_same_length(x, y):
    """Pad x and y so that the results have the same length (second dimension)."""
    x_length = tf.shape(x)[1]
    y_length = tf.shape(y)[1]

    max_length = tf.maximum(x_length, y_length)

    x = tf.pad(x, [[0, 0], [0, max_length - x_length]])
    y = tf.pad(y, [[0, 0], [0, max_length - y_length]])
    return x, y, max_length.numpy()

def pad_tensor(x, max_length):
    
    x_length = tf.shape(x)[0]
    x = tf.pad(x, [[0, max_length - x_length]])

    return x

def save_metrics_to_file(path_details_file, **kwargs): 
    print("Saving metrics to file: ", path_details_file)
    with open(path_details_file, "a", encoding="utf-8") as fic: 
        for k, v in kwargs.items():
            if k == "Date":
                fic.write(f"##########################################################\n")
            fic.write(f"{k} = {str(v)}\n")

def save_results_to_file(path_details_file, input_lines, output_lines, predictions): 
   
    with open(path_details_file, "a", encoding="utf-8") as fic: 
         fic.write(f"<INPUT> {str(input_lines.decode('utf-8'))} <EXPECTED> {str(output_lines.decode('utf-8'))} <PREDICTED> {predictions}\n")
            
def save_config_to_file(path_details_file):
    print("Saving conifg to file: ", path_details_file)
    with open(path_details_file, "w", encoding="utf-8") as fic: 
        for k, v in Config.__dict__.items():
            if k.isupper(): 
                fic.write(f"{k} = {str(v)}\n")

