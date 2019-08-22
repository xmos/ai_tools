# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

import numpy as np

from bitstring import BitArray
from XS3VPU import XS3VPU
from test_XS3VPU_utils import strictfail, vpu
from test_XS3VPU_utils import BPV, VAC, VALID_BPES, INVALID_BPES, VALID_DTYPES


class Test_XS3VPU_Constructor_Args():

    @pytest.mark.parametrize("bpv", [BPV, pytest.param(128, marks=strictfail)])
    @pytest.mark.parametrize("bpe", VALID_BPES + INVALID_BPES)
    @pytest.mark.parametrize("vac", [VAC, pytest.param(16, marks=strictfail)])
    def test_permitted_constants(self, bpv, bpe, vac):
        XS3VPU(bpe=bpe, bpv=bpv, vac=vac)


@pytest.mark.parametrize("bpe", VALID_BPES)
class Test_XS3VPU_Constructor():

    def test_internal_data_types(self, vpu, bpe):
        assert vpu._single == VALID_DTYPES[bpe]['single']
        assert vpu._double == VALID_DTYPES[bpe]['double']
        assert vpu._quad == VALID_DTYPES[bpe]['quad']

    def test_register_init(self, vpu, bpe):
        assert vpu._vC.shape == vpu._vD.shape == vpu._vR.shape == (BPV//bpe,)
        assert vpu._vC.dtype == vpu._vD.dtype == vpu._vR.dtype == VALID_DTYPES[bpe]['single']

    def test_masks(self, vpu, bpe):
        assert vpu._lowmask.dtype == vpu._double
        assert BitArray(int=int(vpu._lowmask), length=bpe*2).bin == '0'*bpe + '1'*bpe

    def test_vector_masks(self, vpu, bpe):
        assert vpu._vec_lowmask.dtype == vpu._double
        assert vpu._vec_lowmask.shape == vpu.vR.shape
        for j, vl in enumerate(vpu._vec_lowmask):
            assert np.binary_repr(vl, bpe*2) == '0'*bpe + '1'*bpe

    def test_quadmasks(self, vpu, bpe):
        assert vpu._quadmask.dtype == vpu._vec_quadmask.dtype == vpu._quad
        assert vpu._vec_quadmask.shape == vpu.vR.shape

        assert BitArray(int=vpu._quadmask, length=bpe*4).bin == '0'*bpe*2 + '1'*bpe*2
        for j, vl in enumerate(vpu._vec_quadmask):
            assert BitArray(int=vl, length=bpe*4).bin == '0'*bpe*2 + '1'*bpe*2


if __name__ == "__main__":
    pytest.main()
