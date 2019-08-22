# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

import numpy as np

from bitstring import BitArray
from test_XS3VPU_utils import vpu, VALID_BPES

MASK_PATTERNS = ['1001', '0110', '1110']

@pytest.fixture(params=MASK_PATTERNS)
def vh(request, bpe):
    hpattern = request.param
    num_rep = bpe//len(hpattern)
    return BitArray(bin=''.join([hpattern] * num_rep))

@pytest.fixture(params=MASK_PATTERNS)
def vl(request, bpe):
    lpattern = request.param
    num_rep = bpe//len(lpattern)
    return BitArray(bin=''.join([lpattern] * num_rep))


@pytest.mark.parametrize("bpe", VALID_BPES)
class Test_XS3VPU_Combine_Split_Scalars():

    def test_combine_singles(self, vpu, vh, vl, bpe):
        vout = vpu._combine_singles(vpu._single(vh.int), vpu._single(vl.int))

        assert vout.dtype == vpu._double
        assert vout == BitArray(bin=vh.bin + vl.bin).int

    def test_split_double(self, vpu, vh, vl, bpe):
        vin = BitArray(bin=vh.bin + vl.bin)
        vlout, vhout = vpu._split_double(vpu._double(vin.int))

        assert vhout.dtype == vlout.dtype == vpu._single
        assert BitArray(int=int(vhout), length=bpe) == vh
        assert BitArray(int=int(vlout), length=bpe) == vl

    def test_combine_doubles(self, vpu, vh, vl, bpe):
        vh_val = vpu._combine_singles(vpu._single(vh.int), vpu._single(vh.int))
        vl_val = vpu._combine_singles(vpu._single(vl.int), vpu._single(vl.int))
        if bpe == 32:
            with pytest.raises(ValueError):
                vpu._combine_doubles(vh_val, vl_val)
        else:
            vout = vpu._combine_doubles(vh_val, vl_val)
            assert vout.dtype == vpu._quad

            high_bits = BitArray(int=int(vh_val), length=bpe*2)
            low_bits = BitArray(int=int(vl_val), length=bpe*2)
            assert vout == BitArray(bin=high_bits.bin + low_bits.bin).int

    def test_split_quad(self, vpu, vh, vl, bpe):
        vh_val = vpu._combine_singles(vpu._single(vh.int), vpu._single(vh.int))
        vl_val = vpu._combine_singles(vpu._single(vl.int), vpu._single(vl.int))

        high_bits = BitArray(int=int(vh_val), length=bpe*2)
        low_bits = BitArray(int=int(vl_val), length=bpe*2)
        vin = BitArray(bin=high_bits.bin + low_bits.bin)

        if bpe == 32:
            with pytest.raises(ValueError):
                vpu._split_quad(vpu._quad(vin.int))
        else:
            vlout, vhout = vpu._split_quad(vpu._quad(vin.int))

            assert vhout.dtype == vlout.dtype == vpu._double
            assert BitArray(int=int(vhout), length=2*bpe) == high_bits
            assert BitArray(int=int(vlout), length=2*bpe) == low_bits


class Test_XS3VPU_Combine_Split_Registers():
    @pytest.mark.parametrize("bpe", [16, 32])
    def test_combine_vD_vR_16_32(self, vpu, vh, vl, bpe):
        vpu.VLDD(np.array([vpu._single(vh.int)] * vpu._ve))
        vpu.VLDR(np.array([vpu._single(vl.int)] * vpu._ve))
        vexp = np.array(
            [vpu._double(BitArray(bin=vh.bin + vl.bin).int)] * vpu._ve)
        vout = vpu._combine_vD_vR()

        assert vout.dtype == vpu._double
        assert vout.shape == vpu.vR.shape
        assert np.all(vout == vexp)

    @pytest.mark.parametrize("bpe", [8])
    def test_combine_vD_vR_8_case1(self, vpu, vh, vl, bpe):
        vin = np.array([vpu._single(vl.int), vpu._single(vh.int)] * (vpu._ve//2))
        vpu.VLDD(vin)
        vpu.VLDR(vin)
        vexp = np.array(
            [vpu._quad(BitArray(bin=(vh.bin + vl.bin) * 2).int)] * (vpu._ve//2))
        vout = vpu._combine_vD_vR()

        assert vout.dtype == vpu._quad
        assert vout.shape == (vpu.vR.shape[0]//2,)
        assert np.all(vout == vexp)

    @pytest.mark.parametrize("bpe", [8])
    def test_combine_vD_vR_8_case2(self, vpu, vh, vl, bpe):
        vpu.VLDD(np.array([vpu._single(vh.int)] * vpu._ve))
        vpu.VLDR(np.array([vpu._single(vl.int)] * vpu._ve))
        vexp = np.array(
            [vpu._quad(BitArray(bin=vh.bin * 2 + vl.bin * 2).int)] * (vpu._ve//2))
        vout = vpu._combine_vD_vR()

        assert vout.dtype == vpu._quad
        assert vout.shape == (vpu.vR.shape[0]//2,)
        assert np.all(vout == vexp)

    @pytest.mark.parametrize("bpe", [16, 32])
    def test_split_to_vD_vR_16_32(self, vpu, vh, vl, bpe):
        vH = np.array([vpu._single(vh.int)] * vpu._ve)
        vL = np.array([vpu._single(vl.int)] * vpu._ve)
        vpu.VLDD(vH)
        vpu.VLDR(vL)
        vcombined = vpu._combine_vD_vR()
        vpu.VCLRDR()
        vpu._split_to_vD_vR(vcombined)

        assert vpu._vD.shape == vpu._vR.shape == vH.shape == vL.shape
        assert vpu._vD.dtype == vpu._vR.dtype == vH.dtype == vL.dtype
        assert np.all(vpu._vD == vH)
        assert np.all(vpu._vR == vL)

    @pytest.mark.parametrize("bpe", [8])
    def test_split_to_vD_vR_8_case1(self, vpu, vh, vl, bpe):
        vin = np.array([vpu._single(vl.int), vpu._single(vh.int)] * (vpu._ve//2))
        vpu.VLDD(vin)
        vpu.VLDR(vin)
        vcombined = vpu._combine_vD_vR()
        vpu.VCLRDR()
        vpu._split_to_vD_vR(vcombined)

        assert vpu._vD.shape == vpu._vR.shape == vin.shape
        assert vpu._vD.dtype == vpu._vR.dtype == vin.dtype
        assert np.all(vpu._vD == vin)
        assert np.all(vpu._vR == vin)

    @pytest.mark.parametrize("bpe", [8])
    def test_split_to_vD_vR_8_case2(self, vpu, vh, vl, bpe):
        vH = np.array([vpu._single(vh.int)] * vpu._ve)
        vL = np.array([vpu._single(vl.int)] * vpu._ve)
        vpu.VLDD(vH)
        vpu.VLDR(vL)
        vcombined = vpu._combine_vD_vR()
        vpu.VCLRDR()
        vpu._split_to_vD_vR(vcombined)

        assert vpu._vD.shape == vpu._vR.shape == vH.shape == vL.shape
        assert vpu._vD.dtype == vpu._vR.dtype == vH.dtype == vL.dtype
        assert np.all(vpu._vD == vH)
        assert np.all(vpu._vR == vL)



if __name__ == "__main__":
    pytest.main()
