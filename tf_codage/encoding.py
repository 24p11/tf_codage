from collections import Counter
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Encode multi-label targets by their frequency rank."""

    def __init__(self, unique=False, sort_labels=False, unknown_label="UNK"):
        self._unique = unique
        self._sort_labels = sort_labels
        self._unknown_label = unknown_label

    def fit(self, y):
        counts = Counter()
        for labels in y:
            counts.update(labels)
        classes_ = [l for l, _ in counts.most_common()] + [self._unknown_label]
        self.classes_ = np.array(classes_)
        self._classes_map = {label: enc for enc, label in enumerate(classes_)}
        self._unknown_id = self._classes_map[self._unknown_label]
        return self

    def _encode(self, labels):
        indices = [self._classes_map.get(c, self._unknown_id) for c in labels]
        if self._unique:
            indices = np.unique(indices)
        if self._sort_labels:
            indices = sorted(indices)
        return np.asarray(indices)

    def _decode(self, y):
        return self.classes_[np.array(y)]

    def inverse_transform(self, y):
        return np.array([self._decode(y_) for y_ in y])

    def transform(self, X):
        return np.array([self._encode(x) for x in X])
