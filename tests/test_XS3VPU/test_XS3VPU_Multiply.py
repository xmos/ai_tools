# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

import numpy as np

from test_XS3VPU_utils import vpu, VALID_BPES


@pytest.mark.parametrize("bpe", [8])
class Test_XS3VPU_VLMACC_8bit():

    def test_VLMACC_8bit_low(self, vpu, bpe):
        vin1 = np.linspace(0, 2, vpu._ve, dtype=vpu._single)
        vin2 = np.flip(np.copy(vin1))
        vpu.VLDC(vin1)

        for j in range(3):
            vin2[1::2] = j
            vpu.VLMACC(vin2)
            assert np.all(vpu._vR[::2] == (vin1 * vin2 * (j+1))[::2])
            assert np.all(vpu._vR[1::2] == np.zeros(vpu._ve//2, dtype=vpu._single))
            assert np.all(vpu.vD == np.zeros(vpu._ve, dtype=vpu._single))
            assert np.all(vpu._combine_vD_vR() == (vin1 * vin2 * (j+1))[::2])

    def test_VLMACC_8bit_high_low(self, vpu, bpe):
        vin = np.ones(vpu._ve, dtype=vpu._single)
        vpu.VLDC(vin)
        filler = vpu._single(0xff) * np.ones(vpu._ve, dtype=vpu._single)
        vpu.VLDR(filler)
        vpu.VLMACC(vin)

        assert np.all(vpu.vD[::2] == vin[::2])
        assert np.all(vpu.vD[1::2] == np.zeros(vpu._ve//2, dtype=vpu._single))
        assert np.all(vpu.vR == np.zeros(vpu._ve, dtype=vpu._single))
        assert np.all(vpu._combine_vD_vR() == (vpu._quad(vin) << vpu._quad(16))[::2])

    def test_VLMACC_8bit_high_high(self, vpu, bpe):
        vin = np.ones(vpu._ve, dtype=vpu._single)
        vpu.VLDC(vin)
        filler = vpu._single(0xff) * np.ones(vpu._ve, dtype=vpu._single)
        vpu.VLDR(filler)
        filler[1::2] = np.single(0)
        vpu.VLDD(filler)
        vpu.VLMACC(vin)

        assert np.all(vpu.vD[1::2] == vin[::2])
        assert np.all(vpu.vD[::2] == np.zeros(vpu._ve//2, dtype=vpu._single))
        assert np.all(vpu.vR == np.zeros(vpu._ve, dtype=vpu._single))
        assert np.all(vpu._combine_vD_vR() == (vpu._quad(vin) << vpu._quad(24))[::2])

    def test_VLMACC_8bit_neg(self, vpu, bpe):
        vin = vpu._single(-2**6) * np.ones(vpu._ve, dtype=vpu._single)
        vexp = (vpu._double(vin) * vpu._double(vin))[::2]
        vpu.VLDC(vin)
        vpu.VLMACC(vin)

        assert np.all(vpu._combine_vD_vR() == vexp)


@pytest.mark.parametrize("bpe", [16])
class Test_XS3VPU_VLMACC_16bit():

    def test_VLMACC_16bit_low(self, vpu, bpe):
        vin1 = np.arange(vpu._ve, dtype=vpu._single)
        vin2 = np.flip(vin1)
        vpu.VLDC(vin1)

        for j in range(10):
            vpu.VLMACC(vin2)
            assert np.all(vpu.vR == (vin1*vin2*(j+1)))
            assert np.all(vpu.vD == np.zeros(vpu._ve, dtype=vpu._single))
            assert np.all(vpu._combine_vD_vR() == (vin1*vin2*(j+1)))

    def test_VLMACC_16bit_high(self, vpu, bpe):
        vin = np.arange(vpu._ve, dtype=vpu._single)
        vshift = vpu._single(2**8) * np.ones(vpu._ve, dtype=vpu._single)
        vpu.VLDC(vin * vshift)
        vpu.VLMACC(vshift)

        assert np.all(vpu.vD == vin)
        assert np.all(vpu.vR == np.zeros(vpu._ve, dtype=vpu._single))
        assert np.all(vpu._combine_vD_vR() == (vpu._double(vin) << vpu._double(16)))

    def test_VLMACC_16bit_neg(self, vpu, bpe):
        vin = vpu._single(-2**15) * np.ones(vpu._ve, dtype=vpu._single)
        vexp = (vpu._double(vin) * vpu._double(vin))
        vpu.VLDC(vin)
        vpu.VLMACC(vin)

        assert np.all(vpu._combine_vD_vR() == vexp)


@pytest.mark.parametrize("bpe", [8])
class Test_XS3VPU_VLMACCR_8bit():

    def test_VLMACCR_8bit_no_bias(self, vpu, bpe):
        a, b = 15, 11
        vpu.VLDC(vpu._single(a) * np.ones(vpu._ve, dtype=vpu._single))

        vexp = np.zeros(vpu._ve//2, dtype=vpu._quad)
        for j in range(vpu._ve+7):
            vexp[j % (vpu._ve//2)] += vpu._quad(vpu._ve*a*b)
            vpu.VLMACCR(vpu._single(b) * np.ones(vpu._ve, dtype=vpu._single))
            assert np.all(vpu._combine_vD_vR() == vexp)

    def test_VLMACCR_8bit_low_bias(self, vpu, bpe):
        vD_vR = np.linspace(1, vpu._ve//2, vpu._ve//2, dtype=vpu._quad)
        vpu._split_to_vD_vR(vD_vR)
        a, b = 3, 7
        vpu.VLDC(vpu._single(a) * np.ones(vpu._ve, dtype=vpu._single))

        vexp = np.copy(vD_vR)
        for j in range(vpu._ve+5):
            vexp = np.hstack([vexp[-1] + vpu._quad(vpu._ve*a*b), vexp[:-1]])
            vpu.VLMACCR(vpu._single(b) * np.ones(vpu._ve, dtype=vpu._single))
            assert np.all(vpu._combine_vD_vR() == vexp)

    def test_VLMACCR_8bit_high_bias(self, vpu, bpe):
        vD_vR = np.linspace(1, vpu._ve//2, vpu._ve//2, dtype=vpu._quad) << vpu._quad(16)
        vpu._split_to_vD_vR(vD_vR)
        a, b = 19, 72
        vpu.VLDC(vpu._single(a) * np.ones(vpu._ve, dtype=vpu._single))

        vexp = np.copy(vD_vR)
        for j in range(vpu._ve+12):
            vexp = np.hstack([vexp[-1] + vpu._quad(vpu._ve*a*b), vexp[:-1]])
            vpu.VLMACCR(vpu._single(b) * np.ones(vpu._ve, dtype=vpu._single))
            assert np.all(vpu._combine_vD_vR() == vexp)


@pytest.mark.parametrize("bpe", [16])
class Test_XS3VPU_VLMACCR_16bit():

    def test_VLMACCR_16bit_no_bias(self, vpu, bpe):
        a, b = 251, 129
        vpu.VLDC(vpu._single(a) * np.ones(vpu._ve, dtype=vpu._single))

        vexp = np.zeros(vpu._ve, dtype=vpu._quad)
        for j in range(vpu._ve+25):
            vexp[j % vpu._ve] += vpu._quad(vpu._ve*a*b)
            vpu.VLMACCR(vpu._single(b) * np.ones(vpu._ve, dtype=vpu._single))
            assert np.all(vpu._combine_vD_vR() == vexp)

    def test_VLMACCR_16bit_low_bias(self, vpu, bpe):
        vD_vR = np.linspace(1+700, vpu._ve+700, vpu._ve, dtype=vpu._double)
        vpu._split_to_vD_vR(vD_vR)
        a, b = -761, 213
        vpu.VLDC(vpu._single(a) * np.ones(vpu._ve, dtype=vpu._single))

        vexp = np.copy(vD_vR)
        for j in range(vpu._ve+79):
            vexp = np.hstack([vexp[-1] + vpu._quad(vpu._ve*a*b), vexp[:-1]])
            vpu.VLMACCR(vpu._single(b) * np.ones(vpu._ve, dtype=vpu._single))
            assert np.all(vpu._combine_vD_vR() == vexp)

    def test_VLMACCR_16bit_high_bias(self, vpu, bpe):
        vD_vR = np.linspace(1-1278, vpu._ve-1278, vpu._ve, dtype=vpu._double) << vpu._double(16)
        vpu._split_to_vD_vR(vD_vR)
        a, b = 897, -2357
        vpu.VLDC(vpu._single(a) * np.ones(vpu._ve, dtype=vpu._single))

        vexp = np.copy(vD_vR)
        for j in range(vpu._ve+5):
            vexp = np.hstack([vexp[-1] + vpu._quad(vpu._ve*a*b), vexp[:-1]])
            vpu.VLMACCR(vpu._single(b) * np.ones(vpu._ve, dtype=vpu._single))
            assert np.all(vpu._combine_vD_vR() == vexp)

    def test_VLMACCR_16bit_sat(self, vpu, bpe):
        vD_vR = np.linspace(1-1278, vpu._ve-1278, vpu._ve, dtype=vpu._double) << vpu._double(16)
        vpu._split_to_vD_vR(vD_vR)
        a, b = 897, -2357
        vpu.VLDC(vpu._single(a) * np.ones(vpu._ve, dtype=vpu._single))

        vexp = np.copy(vD_vR)
        for j in range(vpu._ve+5):
            vexp = np.hstack([vexp[-1] + vpu._quad(vpu._ve*a*b), vexp[:-1]])
            vpu.VLMACCR(vpu._single(b) * np.ones(vpu._ve, dtype=vpu._single))
            assert np.all(vpu._combine_vD_vR() == vexp)


if __name__ == "__main__":
    pytest.main()
