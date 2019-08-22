# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

import numpy as np

from test_XS3VPU_utils import vpu, VALID_BPES


@pytest.mark.parametrize("bpe", VALID_BPES)
class Test_XS3VPU_VLADD():

    def test_VLADD(self, vpu):
        smin, smax = vpu._sat_bounds(vpu._single)
        vin = np.linspace(smin, smax, vpu._ve, dtype=vpu._single)
        vpu.VLADD(vin)
        assert np.all(vpu.vR == vin)
        vpu.VLADD(np.flip(vin))
        assert np.all(vpu.vR == np.zeros(shape=(vpu._ve), dtype=vpu._single))

    def test_VLADD_sat_max(self, vpu):
        _, smax = vpu._sat_bounds(vpu._single)
        vin = np.array([smax]*vpu._ve)
        vpu.VLADD(vin)
        assert np.all(vpu.vR == vin)
        vpu.VLADD(vin)
        assert np.all(vpu.vR == vin)
        
    def test_VLADD_sat_min(self, vpu):
        smin, _ = vpu._sat_bounds(vpu._single)
        vin = np.array([smin]*vpu._ve)
        vpu.VLADD(vin)
        assert np.all(vpu.vR == vin)
        vpu.VLADD(vin)
        assert np.all(vpu.vR == vin)


@pytest.mark.parametrize("bpe", VALID_BPES)
class Test_XS3VPU_VLSUB():

    def test_VLSUB(self, vpu):
        smin, smax = vpu._sat_bounds(vpu._single)
        vin = np.linspace(smin, smax, vpu._ve, dtype=vpu._single)
        vpu.VLSUB(vin)
        assert np.all(vpu.vR == vin)
        vpu.VLSUB(vin)
        assert np.all(vpu.vR == np.zeros(shape=(vpu._ve), dtype=vpu._single))

    def test_VLSUB_sat_min(self, vpu):
        _, smax = vpu._sat_bounds(vpu._single)
        vin = np.array([smax]*vpu._ve)
        vpu.VLSUB(vin)
        assert np.all(vpu.vR == vin)
        vpu.VLSUB(-vin)
        assert np.all(vpu.vR == -vin)

    def test_VLSUB_sat_max(self, vpu):
        _, smax = vpu._sat_bounds(vpu._single)
        vin = np.array([smax]*vpu._ve)
        vpu.VLSUB(-vin)
        assert np.all(vpu.vR == -vin)
        vpu.VLSUB(vin)
        assert np.all(vpu.vR == vin)


if __name__ == "__main__":
    pytest.main()
