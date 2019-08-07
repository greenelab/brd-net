
import os
import sys

import numpy as np
import pandas
import pytest

brdnet_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(brdnet_path + '/../brdnet')
import utils

@pytest.mark.parametrize('pred_Y, true_Y, correct_answer',
                         [([0, 1, 1, 0], [1, 0, 0, 1], 0),
                          ([0, 1, 0, 1], [1, 0, 1, 1], .25),
                          ([0, 1, 0, 1], [0, 0, 1, 1], .5),
                          ([0, 1, 0, 1], [0, 1, 1, 1], .75),
                          ([0, 1, 0, 1], [0, 1, 0, 1], 1),
                         ]
                        )
def test_calculate_accuracy(pred_Y, true_Y, correct_answer):
    assert utils.calculate_accuracy(pred_Y, true_Y) == pytest.approx(correct_answer)
