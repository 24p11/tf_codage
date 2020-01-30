from tf_codage import data

def test_multilabel_dataset():
    "test importing multilabel data into dataframe"
    df, num_labels = data.make_multilabel_dataframe("tests/cro_data.csv", min_examples=0)
    assert len(df) == 2
    assert num_labels == 3
    assert df.target[0] == {'LDKA900'}
    assert df.target[1] == {'AELB001','ZZLP025'}