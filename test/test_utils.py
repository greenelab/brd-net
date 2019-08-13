
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

@pytest.mark.parametrize('Y, true_percent',
                         [([1, 2, 2, 2, 3], .6),
                          ([0, 0, 1, 1], .5),
                          ([0, 0, 0, 0, 1, 1], 4/6),
                          ([0, 0, 0, 0], 1),
                          ([1, 2, 3, 4], .25),
                         ]
                        )
def test_get_larger_class_percentage(Y, true_percent):
    assert utils.get_larger_class_percentage(Y) == pytest.approx(true_percent)

