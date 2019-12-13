# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import numpy as np


def compare_tensor_files(expected_file, expected_quantization, predicted_file,
                         predicted_quantization, abs_tol=0.001):
    expected_values = np.fromfile(expected_file, dtype='int8') 
    expected_zero_point = expected_quantization.get('zero_point', 0.0)
    expected_scale = expected_quantization.get('scale', 1.0)
    predicted_values = np.fromfile(predicted_file, dtype='int8')
    predicted_zero_point = predicted_quantization.get('zero_point', 0.0)
    predicted_scale = predicted_quantization.get('scale', 1.0)

    len_ratio = len(expected_values) / len(predicted_values)
    if len_ratio == 2:
        predicted_values = np.fromfile(predicted_file, dtype='int16')
    elif len_ratio == 4:
        predicted_values = np.fromfile(predicted_file, dtype='int32')

    dequantized_expected_values = (expected_values - expected_zero_point) * expected_scale
    dequantized_predicted_values = (predicted_values - predicted_zero_point) * predicted_scale

    for ev, pv in zip(dequantized_expected_values, dequantized_predicted_values):
        if abs(ev-pv) > abs_tol:
            print(ev, pv, abs(ev-pv))
            return False

    return True
