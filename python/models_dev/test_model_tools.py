# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import model_tools as mt
import numpy as np


def test_get_mnist():
    k = [(2, 0, 1, 0, 1, 0),
         (2, 0, 1, 1, 1, 0),
         (2, 0, 0, 1, 1, 0),
         (2, 1, 0, 0, 1, 0),
         (2, 0, 0, 0, 0, 0),
         (2, 0, 0, 0, 1, 1),
         (0, 0, 0, 0, 1, 0),
         (1, 0, 0, 0, 1, 0),
         (2, 0, 0, 0, 1, 0),
         (3, 0, 0, 0, 1, 0),
         (2, 0, 1, 1, 1, 0),
         (2, 1, 1, 0, 1, 0)]
    for i, l in enumerate(k):
        if l[2]:
            x_train, x_test, x_val, y_train, y_test, y_val = mt.get_mnist(*l)
            assert len(y_train) == 48000
            assert len(x_test) == 10000
            assert len(x_val) == 12000
        else:
            x_train, x_test, y_train, y_test = mt.get_mnist(*l)
            assert len(y_train) == 60000
            assert len(x_test) == 10000
        assert type(x_train) == np.ndarray
        if i == 0:
            assert x_train.shape == (48000, 32, 32, 1)
            assert y_train[0] == 7
        if i == 1:
            assert x_train.shape == (48000, 1024)
            assert y_train[0] == 7
        if i == 2:
            assert x_train.shape == (60000, 1024)
            assert y_train[0] == 5
        if i == 3:
            assert x_train.shape == (60000, 32, 32, 1)
            assert y_train.shape == (60000, 1, 1, 10)
            assert np.argmax(y_train[0]) == 5
        if i == 4:
            assert x_train.shape == (60000, 32, 32, 1)
            assert y_train[0] == 5
        if i == 5:
            assert x_train.shape == (60000, 32, 32, 1)
            assert type(y_train[0]) == np.float32
            assert y_train[0] == 5.0
        if i == 6:
            assert x_train.shape == (60000, 28, 28, 1)
        if i == 7:
            assert x_train.shape == (60000, 30, 30, 1)
        if i == 8:
            assert x_train.shape == (60000, 32, 32, 1)
        if i == 9:
            assert x_train.shape == (60000, 34, 34, 1)
        if i == 10:
            assert x_train.shape == (48000, 1024)
        if i == 11:
            assert x_train.shape == (48000, 32, 32, 1)
            assert y_train.shape == (48000, 1, 1, 10)
