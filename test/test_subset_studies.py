
import os
import sys

import numpy as np
import pandas
import pytest

brdnet_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(brdnet_path + '/../brdnet')
import subset_studies

seven_columns = ['SRP{}.xxxx'.format(i) for i in range(7)]
three_columns = ['SRP{}.xxxx'.format(i) for i in range(7,10)]

seven_df = pandas.DataFrame(np.ones((3,7)), columns=seven_columns)
three_df = pandas.DataFrame(np.ones((3,3)), columns=three_columns)

@pytest.mark.parametrize('df1, df2, fraction',
                         [
                          (seven_df, three_df, .5),
                          (three_df, seven_df, .5),
                          (three_df, seven_df, .3),
                          (three_df, seven_df, .7),
                          (seven_df, three_df, .3),
                          ])
def test_rebalance(df1, df2, fraction):
    balanced_df1, balanced_df2 = subset_studies.rebalance(df1, df2, fraction)
    print(balanced_df1)

    total_samples = len(balanced_df1.columns) + len(balanced_df2.columns)

    assert len(balanced_df1.columns) == pytest.approx(total_samples * fraction)
