# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

import numpy as np

from test_XS3VPU_utils import vpu, VALID_BPES, VALID_BOUNDS, VALID_DTYPES, Int

@pytest.mark.parametrize("bpe", VALID_BPES)
class Test_XS3VPU_Sat_Bounds():

    @pytest.mark.parametrize("mode", ['single', 'double', 'quad'])
    def test_sat_bounds(self, vpu, bpe, mode):
        dtype = VALID_DTYPES[bpe][mode]
        bmin, bmax = vpu._sat_bounds(dtype)
        assert bmin.dtype == bmax.dtype == dtype
        assert bmin, bmax  == VALID_BOUNDS[bpe][dtype]


@pytest.mark.parametrize("bpe", VALID_BPES)
class Test_XS3VPU_ssat():

    def test_ssat_min(self, vpu, bpe):
        smin, dmin = np.iinfo(vpu._single).min, np.iinfo(vpu._double).min
        vin = np.linspace(dmin, smin, vpu._ve, dtype=vpu._double)
        vout = vpu._ssat(vin)

        expected_min = VALID_BOUNDS[bpe]['single'][0]
        vexp = np.array([expected_min] * vin.shape[0])

        assert vout.dtype == vpu._single
        assert np.all(vout == vexp)
        
    def test_ssat_max(self, vpu, bpe):
        smax, dmax = np.iinfo(vpu._single).max, np.iinfo(vpu._double).max
        vin = np.linspace(smax, dmax, vpu._ve, dtype=vpu._double)
        if vpu._double == np.int64:
            # np bug causes the last element to wrap in 32-bit mode
            vin[-1] = dmax  
        vout = vpu._ssat(vin)

        expected_max = VALID_BOUNDS[bpe]['single'][1]
        vexp = np.array([expected_max] * vin.shape[0])

        assert vout.dtype == vpu._single
        assert np.all(vout == vexp)

    def test_ssat_mid(self, vpu):
        smin, smax = np.iinfo(vpu._single).min, np.iinfo(vpu._single).max
        vin = np.linspace(smin + 1, smax, vpu._ve, dtype=vpu._double)
        vout = vpu._ssat(vin)

        assert vout.dtype == vpu._single
        assert np.all(vout == vin)


@pytest.mark.parametrize("bpe", VALID_BPES)
class Test_XS3VPU_dsat():

    def test_dsat_min(self, vpu, bpe):
        dmin = np.iinfo(vpu._double).min
        if bpe == 32:
            qmin = -2**(128-1)
        else:
            qmin = np.iinfo(vpu._quad).min
        vin = np.linspace(qmin, dmin, vpu._ve, dtype=vpu._quad)
        vout = vpu._dsat(vin)

        expected_min = VALID_BOUNDS[bpe]['double'][0]
        vexp = np.array([expected_min] * vin.shape[0])

        assert vout.dtype == vpu._double
        assert np.all(vout == vexp)

    def test_dsat_max(self, vpu, bpe):
        dmax = np.iinfo(vpu._double).max
        if bpe == 32:
            qmax = 2**(128-1) - 1
        else:
            qmax = np.iinfo(vpu._quad).max
        vin = np.linspace(dmax, qmax, vpu._ve, dtype=vpu._quad)
        if vpu._quad == np.int64:
            # np bug causes the last element to wrap in 16-bit mode
            vin[-1] = qmax
        vout = vpu._dsat(vin)

        expected_max = VALID_BOUNDS[bpe]['double'][1]
        vexp = np.array([expected_max] * vin.shape[0])

        assert vout.dtype == vpu._double
        assert np.all(vout == vexp)

    def test_dsat_mid(self, vpu, bpe):
        dmin, dmax = np.iinfo(vpu._double).min, np.iinfo(vpu._double).max
        if bpe == 32:
            # np bug causes the first and last element to wrap in 32-bit mode
            vin = np.linspace(dmin + 1, dmax, vpu._ve, dtype=vpu._double)
            vin[0], vin[-1] = dmin + 1, dmax
            vin = np.array([int(v) for v in vin], dtype=vpu._quad)
        else:
            vin = np.linspace(dmin + 1, dmax, vpu._ve, dtype=vpu._quad)
        vout = vpu._dsat(vin)

        assert vout.dtype == vpu._double
        assert np.all(vout == vin)


@pytest.mark.parametrize("bpe", [8])
class Test_XS3VPU_qsat_8bit():

    def test_qsat_min(self, vpu, bpe):
        qmin, hmin = np.iinfo(vpu._quad).min, -2**(128-1)
        vin = list(range(hmin, qmin, round(-hmin+qmin)//(vpu._ve//2)))
        vin = np.array(vin, dtype=Int)
        vout = vpu._qsat(vin)

        expected_min = VALID_BOUNDS[bpe]['quad'][0]
        vexp = np.array([expected_min] * vin.shape[0])

        assert vout.dtype == vpu._quad
        assert np.all(vout == vexp)

    def test_qsat_max(self, vpu, bpe):
        qmax, hmax = np.iinfo(vpu._quad).max, 2**(128-1) - 1
        vin = list(range(qmax, hmax, round(hmax-qmax)//(vpu._ve//2)))
        vin = np.array(vin, dtype=Int)
        vout = vpu._qsat(vin)

        expected_max = VALID_BOUNDS[bpe]['quad'][1]
        vexp = np.array([expected_max] * vin.shape[0])

        assert vout.dtype == vpu._quad
        assert np.all(vout == vexp)

    def test_dsat_mid(self, vpu):
        qmin, qmax = np.iinfo(vpu._quad).min, np.iinfo(vpu._quad).max
        vin = np.linspace(qmin + 1, qmax, vpu._ve, dtype=vpu._quad)
        vout = vpu._qsat(vin)

        assert vout.dtype == vpu._quad
        assert np.all(vout == vin)


if __name__ == "__main__":
    pytest.main()
