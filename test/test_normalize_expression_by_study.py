import os
import sys

import numpy
import pandas
import pytest

brdnet_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(brdnet_path + '/../brdnet')
import normalize_expression_by_study


@pytest.mark.parametrize('study, std',
                         [('study_one', 1),
                          ('study_two', 0),
                          ('study_three', 0),
                          ('study_four', 0),
                          ('study_five', 1),
                          ]
                         )
def test_normalize_study(study, std):
    data = {'study_one.a': [1, 1, 1],
            'study_one.b': [2, 3, 4],
            'study_two.asdf': [1, 2, 2],
            'study_three.me': [0, 0, 0],
            'study_three.mo': [0, 0, 0],
            'study_four.': [0, 0, 0],
            'study_five.SRR013949': [5, 5, 5],
            'study_five.SRR013939': [1, 3, 2],
            'study_five.SRR013919': [5, 5, 5],
            'study_five.SRR013929': [-1, -8, -3],
            }
    df = pandas.DataFrame(data)

    normalized_df = normalize_expression_by_study.normalize_study(df, study)

    # Test whether all rows have a zero mean
    means = normalized_df.mean(axis=1)
    assert means.eq(0).all()

    # Test whether all rows have the correct standard deviation
    stds = normalized_df.std(axis=1, ddof=0)
    if numpy.isnan(std):
        assert numpy.isnan(stds).all()
    else:
        assert numpy.isclose(stds, std).all()


def test_normalize_by_study():
    data = {'study_one.a': [1, 1, 1],
            'study_one.b': [-1, -1, -1],
            'study_five.SRR013949': [5, 4, 3],
            'study_five.SRR013939': [-5, -4, -3],
            }
    df = pandas.DataFrame(data)

    target_data = {'study_one.a': [1, 1, 1],
                   'study_one.b': [-1, -1, -1],
                   'study_five.SRR013949': [1, 1, 1],
                   'study_five.SRR013939': [-1, -1, -1],
                   }
    target_df = pandas.DataFrame(target_data).astype('float64')

    normalized_df = normalize_expression_by_study.normalize_by_study(df)

    for column in target_df.columns:
        assert target_df[column].equals(normalized_df[column])
