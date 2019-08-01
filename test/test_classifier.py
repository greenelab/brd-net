import os
import sys

import numpy as np
import pandas
import pytest

brdnet_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(brdnet_path + '/../brdnet')
import classifier


def create_testing_dataframe():
    data = {'0.a': np.zeros(4),
            '0.b': np.zeros(4),
            '1.a': np.ones(4),
            '1.b': np.ones(4),
            '2.a': np.ones(4) * 2,
            '2.b': np.ones(4) * 2,
            '3.a': np.ones(4) * 3,
            '3.b': np.ones(4) * 3,
            '4.a': np.ones(4) * 4,
            '4.b': np.ones(4) * 4,
           }
    df = pandas.DataFrame(data)

    return df

def test_get_validation_set():
    df = create_testing_dataframe()

    train_df, val_df = classifier.get_validation_set(df, validation_fraction=.2)

    assert len(val_df.columns) == 2
    assert len(train_df.columns) == 8
