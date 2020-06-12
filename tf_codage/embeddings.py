from collections import defaultdict
import sys

from itertools import zip_longest
from tqdm import tqdm
from tf_codage.utils import TeeStream

from itertools import groupby

import tensorflow as tf
import numpy as np
import joblib
from pathlib import Path


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

        
        
def generator_to_dataset(generator_factory, batch_size):
    dataset = tf.data.Dataset.from_generator(
        generator_factory,
        output_types=(
            {'input_ids': tf.int32,
             'token_type_ids': tf.int32,
             'attention_mask': tf.int32},
             tf.int32),
        output_shapes=(
            {"input_ids": [510],
             "token_type_ids": [510],
             "attention_mask": [510]}, []))

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(1)
    return dataset
    
class SentenceEmbedder:
    
    SPLIT_SIZE = 10000
    EMBEDDING_BATCH_SIZE = 64
    MAX_SENTENCES = 510
    
    def __init__(self, sentence_tokenizer, word_tokenizer, model):
        self._model = model
        self.sentence_tokenizer = sentence_tokenizer
        self.word_tokenizer = word_tokenizer
        
    def _calculate_embeddings(self, dataset):
        
        model = self.model

        embeddings = []
        for inputs, text_ids in dataset:
            outputs = model(inputs).numpy()
            embeddings += list(zip(outputs, text_ids.numpy()))

        text_groups = groupby(embeddings, key=lambda x: x[1])

        for text_id, sentences_ids in text_groups:
            sentences = list(map(lambda x: x[0], sentences_ids))
            text_embeddings = np.vstack(sentences)
            yield text_embeddings, text_id

    @property
    def model(self):
        return self._model
    
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
        
        def add_progress_bar(generator_factory):
            
            def _generator_with_progress():
                for encoded_text in generator_factory():
                    yield encoded_text
            return _generator_with_progress
        
        for generator in self._split_encode_texts(texts):
            generator = add_progress_bar(generator)
            dataset = generator_to_dataset(generator, self.EMBEDDING_BATCH_SIZE)
            yield dataset
        pbar.close()
    
            
    def _split_encode_texts(self, texts):
        
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
            yield generator
        pbar.close()

    def transform(self, all_texts):

        datasets = self.make_sentence_datasets(all_texts)

        for _, dataset in enumerate(datasets):
            for text_embeddings, text_id in self._calculate_embeddings(dataset):
                yield text_embeddings, all_texts.iloc[text_id]
                
from tf_codage.models import TFCamembertForSentenceEmbedding
class SentenceEmbedderDistributed(SentenceEmbedder):
    """Sentence embedder using dask-distributed to spread task across GPUs on local machine"""
    
    def __init__(self, sentence_tokenizer, word_tokenizer, model_dir):
        self._model_cls = TFCamembertForSentenceEmbedding
        self._model_dir = model_dir
        self.sentence_tokenizer = sentence_tokenizer
        self.word_tokenizer = word_tokenizer

    @staticmethod
    def _run_and_save(model_future, text_group, dataset_id, output_dir=Path('.'), prefix="", batch_size=32):
        
        model = model_future
        
        dataset = generator_to_dataset(text_group.__iter__, batch_size)
 
        data = list(_calculate_embeddings(model, dataset)) 
        output_file = output_dir /  '{}embeddings_{:03d}.joblib'.format(prefix, dataset_id)
        joblib.dump(data, output_file)
 
    def transform_distributed(self, all_texts, output_dir, client):
        
        encoded_texts = self._split_encode_texts(all_texts)
        model_future = client.submit(
            self._model_cls.from_pretrained, self._model_dir)
        
        stream = TeeStream(sys.stdout, sys.__stdout__)
        pbar = tqdm(total=len(all_texts), file=stream)
        
        futures = []
        for j, text_generator in enumerate(encoded_texts):
            client.wait_for_workers(1)
            text_group = list(text_generator())
            pbar.update(len(text_group))
            futures.append(client.submit(self._run_and_save, model_future, text_group, j, output_dir, "", self.EMBEDDING_BATCH_SIZE))
        client.gather(futures)
        pbar.close()