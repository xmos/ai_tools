# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

import numpy as np

from numpy.random import randint
from XS3VPU import XS3VPU

from test_XS3VPU_utils import vpu, VALID_BPES


@pytest.mark.parametrize("bpe", VALID_BPES)
class Test_XS3VPU_VLD_Check():

    def test_VLD_check(self, vpu, bpe):
        vin = randint(-2**(bpe-1), 2**(bpe-1)-1,
                      size=(vpu._ve,), dtype=vpu._single)
        vout = vpu._VLD_check(vin)
        assert np.all(vin == vout)

    def test_VLD_check_copy(self, vpu, bpe):
        vin = np.arange(vpu._ve, dtype=vpu._single)
        vout = vpu._VLD_check(vin)
        for j, v in enumerate(np.flip(np.copy(vin))):
            vin[j] = v
        assert np.all(vin != vout)


@pytest.mark.parametrize("bpe", VALID_BPES)
@pytest.mark.parametrize(("loader", "register"),
                         [(XS3VPU.VLDC, lambda vpu: vpu.vC),
                          (XS3VPU.VLDD, lambda vpu: vpu.vD),
                          (XS3VPU.VLDR, lambda vpu: vpu.vR)])
class Test_XS3VPU_VLD():

    def test_VLD(self, vpu, bpe, loader, register):
        vin = randint(-2**(bpe-1), 2**(bpe-1)-1,
                      size=(vpu._ve,), dtype=vpu._single)
        loader(vpu, vin)
        vout = register(vpu)
        assert np.all(vout == vin)
    
    def test_VLD_copy(self, vpu, bpe, loader, register):
        vin = np.arange(vpu._ve, dtype=vpu._single)
        loader(vpu, vin)
        for j, v in enumerate(np.flip(np.copy(vin))):
            vin[j] = v
        vout = register(vpu)
        assert np.all(vin != vout)


@pytest.mark.parametrize("bpe", VALID_BPES)
class Test_XS3VPU_VCLRDR():

    def test_VCLRDR(self, vpu, bpe):
        vin = np.arange(vpu._ve, dtype=vpu._single)
        vpu.VLDD(vin)
        vpu.VLDR(vin)
        vpu.VCLRDR()
        vexp = np.zeros(vpu._ve, dtype=vpu._single)

        assert vpu._vD.shape == vpu._vR.shape == vexp.shape
        assert vpu._vD.dtype == vpu._vR.dtype == vexp.dtype
        assert np.all(vpu._vD == vexp)
        assert np.all(vpu._vR == vexp)


if __name__ == "__main__":
    pytest.main()
