# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

import numpy as np

from XS3VPU import XS3VPU, Int

strictfail = pytest.mark.xfail(strict=True)

BPV = 256
VAC = 8
VALID_BPES = [8, 16, 32]

INVALID_BPES = [pytest.param(1, marks=strictfail),
                pytest.param(2, marks=strictfail),
                pytest.param(4, marks=strictfail),
                pytest.param(64, marks=strictfail),]

VALID_DTYPES = {8: {'single': np.int8, 'double': np.int16, 'quad': np.int32},
                16: {'single': np.int16, 'double': np.int32, 'quad': np.int64},
                32: {'single': np.int32, 'double': np.int64, 'quad': Int}}

def bnd(num_bits):
    return 2**(num_bits-1) - 1

VALID_BOUNDS = {8:  {'single': (np.int8(-bnd(8)),   np.int8(bnd(8))),
                     'double': (np.int16(-bnd(16)), np.int16(bnd(16))),
                     'quad':   (np.int32(-bnd(32)), np.int32(bnd(32))),},
                16: {'single': (np.int16(-bnd(16)), np.int16(bnd(16))),
                     'double': (np.int32(-bnd(32)), np.int32(bnd(32))),
                     'quad':   (np.int64(-bnd(64)), np.int64(bnd(64))),},
                32: {'single': (np.int32(-bnd(32)), np.int32(bnd(32))),
                     'double': (np.int64(-bnd(64)), np.int64(bnd(64))),
                     'quad':   (int(-bnd(128)),     int(bnd(128))),},
}

@pytest.fixture()
def vpu(bpe):
    return XS3VPU(bpe=bpe, bpv=BPV, vac=VAC)
