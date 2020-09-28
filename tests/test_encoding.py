from tf_codage import encoding
import numpy as np


def test_frequency_encoder():
    labels = [("A", "B"), ("C", "A", "B"), ("A", "B")]
    freq = encoding.FrequencyEncoder(sort_labels=True)
    freq.fit(labels)

    # forward
    assert tuple(freq.transform([("C", "A", "B")])[0]) == (0, 1, 2)
    assert tuple(freq.transform([("C",)])[0]) == (2,)
    assert tuple(freq.transform([("A",)])[0]) == (0,)

    # with unknown class
    assert tuple(freq.transform([("D", "A")])[0]) == (0, 3)

    # inverse
    assert tuple(freq.inverse_transform([(0, 2, 1)])[0]) == ("A", "C", "B")

    # unknown class
    assert tuple(freq.inverse_transform([(3,)])[0]) == ("UNK",)
