from collections import defaultdict
import sys

from itertools import zip_longest
from tqdm import tqdm
from tf_codage.utils import TeeStream

from itertools import groupby

import tensorflow as tf
import numpy as np


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

class SentenceEmbedder:
    
    SPLIT_SIZE = 10000
    EMBEDDING_BATCH_SIZE = 64
    MAX_SENTENCES = 510
    
    def __init__(self, sentence_tokenizer, word_tokenizer, model):
        self.model = model
        self.sentence_tokenizer = sentence_tokenizer
        self.word_tokenizer = word_tokenizer

    def encode(self, txt, txt_id):
        
        sentences = self.sentence_tokenizer.tokenize(txt)[:self.MAX_SENTENCES]
        batch_output = defaultdict(list)
        for sent in sentences:
            token_ids = self.word_tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                return_attention_mask=True,
                max_length=510,
                pad_to_max_length=True,
                return_tensors=None)
            yield token_ids, txt_id

    def make_sentence_datasets(self, texts):
        """Prepare dataset with tokenized sentences and words."""
        
        indexed_texts = zip(texts, range(len(texts)))
        text_groups = grouper(indexed_texts, self.SPLIT_SIZE, fillvalue=("", -1))
        stream = TeeStream(sys.stdout, sys.__stdout__)
        pbar = tqdm(total=len(texts), file=stream)
        def make_generator(gen_texts):

            def generator_factory():
                for text, i in gen_texts:
                    if i > -1:
                        yield from self.encode(text, i)
                    pbar.update(1)

            return generator_factory

        for text_group in text_groups:

            generator = make_generator(text_group)

            dataset = tf.data.Dataset.from_generator(
                generator,
                output_types=(
                    {'input_ids': tf.int32,
                     'token_type_ids': tf.int32,
                     'attention_mask': tf.int32},
                     tf.int32),
                output_shapes=(
                    {"input_ids": [510],
                     "token_type_ids": [510],
                     "attention_mask": [510]}, []))

            dataset = dataset.batch(self.EMBEDDING_BATCH_SIZE, drop_remainder=False)
            dataset = dataset.prefetch(1)
            yield dataset
        pbar.close()


    def transform(self, all_texts):

        datasets = self.make_sentence_datasets(all_texts)

        for j, dataset in enumerate(datasets):
            data = []

            embeddings = []
            for inputs, text_ids in dataset:
                outputs = self.model(inputs).numpy()
                embeddings += list(zip(outputs, text_ids.numpy()))

            text_groups = groupby(embeddings, key=lambda x: x[1])

            for text_id, sentences_ids in text_groups:
                sentences = list(map(lambda x: x[0], sentences_ids))
                text_embeddings = np.vstack(sentences)
                yield text_embeddings, all_texts.iloc[text_id]