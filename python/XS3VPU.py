# Copyright (c) 2019, XMOS Ltd, All rights reserved

import numpy as np

class Shift():
    VEC_SH0 = 0
    VEC_SHL = 1
    VEC_SHR = 2


class Type():  # TODO: use these
    # float types not implemented
    VEC_INT_32 = 0
    VEC_INT_16 = 1
    VEC_INT_8 = 2


class Int(int):
    def __new__(cls, *args, **kwargs):
        return super(Int, cls).__new__(cls, *args, **kwargs)
    @property
    def dtype(self):
        return type(self)


class XS3VPU():

    _DTYPES = {'single': {8: np.int8, 16: np.int16, 32: np.int32},
               'double': {8: np.int16, 16: np.int32, 32: np.int64},
               'quad': {8: np.int32, 16: np.int64, 32: Int},}

    def __init__(self, bpe=8, bpv=256, vac=8):
        # word length settings
        try:
            self._single = self._DTYPES['single'][bpe]
            self._double = self._DTYPES['double'][bpe]
            self._quad = self._DTYPES['quad'][bpe]
        except KeyError:
            raise ValueError("Invalid bpe value, must be in {8, 16, 32}!")
        self._bpe = self._single(bpe)
        self._ve = self._single(bpv//bpe)

        # physical implementation parameters
        if bpv != 256:
            raise ValueError("XS3VPU only supports 256-bit vector length")
        if vac != 8:
            raise ValueError("XS3VPU only supports 8-bit vac")
        self._bpv = self._single(bpv)
        self._vac = self._single(vac)

        # data registers
        self._vC = np.zeros(self._ve, dtype=self._single)
        self._vD = np.zeros(self._ve, dtype=self._single)
        self._vR = np.zeros(self._ve, dtype=self._single)

        # headroom and status registers
        self._vH = self._single(0)
        # TODO: self._vSR

        # bitmasks for combining singles or splitting doubles
        self._lowmask = self._double(2**bpe-1)
        self._vec_lowmask = np.array([self._lowmask]*self._ve)
        self._quadmask = self._quad(
            (self._quad(self._lowmask) << self._quad(self._bpe)) + self._quad(self._lowmask)
        )
        self._vec_quadmask = np.array([self._quadmask]*self._ve, dtype=self._quad)

    @property
    def ve(self):
        return int(self._ve)

    @property
    def acc_period(self):
        if self._bpe == 8:
            return int(self._ve) // 2
        else:
            return int(self._ve)

    def _VLD_check(self, v):
        "Checks if a given vector can be loaded into a register"
        assert v.shape == (self._ve,)
        assert self._single == v.dtype
        return np.copy(v)

    def VLDC(self, val):
        self._vC = self._VLD_check(val)
        
    def VLDD(self, val):
        self._vD = self._VLD_check(val)
        
    def VLDR(self, val):
        self._vR = self._VLD_check(val)

    @property
    def vC(self):
        return np.copy(self._vC)

    @property
    def vD(self):
        return np.copy(self._vD)

    @property
    def vR(self):
        return np.copy(self._vR)

    def VCLRDR(self):
        self._vD.fill(0)
        self._vR.fill(0)

    def VSTC(self):
        raise NotImplementedError()

    def VSTD(self):
        raise NotImplementedError()

    def VSTR(self):
        raise NotImplementedError()

    def _sat_bounds(self, dtype):
        if (self._bpe == 32) and (dtype == self._quad):
            return dtype(-2**(128-1) + 1), dtype(2**(128-1) - 1)
        else:
            return dtype(np.iinfo(dtype).min + 1), dtype(np.iinfo(dtype).max)

    def __sat(self, v, dtype):
        return dtype(np.clip(v, *self._sat_bounds(dtype)))

    def _ssat(self, v):
        return self.__sat(v, self._single)

    def _dsat(self, v):
        return self.__sat(v, self._double)

    def _qsat(self, v):
        if self._bpe != 8:
            raise ValueError("Saturating to quad words is only allowed in 8-bit mode!")
        return self.__sat(v, self._quad)

    def VLADD(self, v):
        self._vR = self._ssat(self._double(self._VLD_check(v)) + self._double(self._vR))

    def VLSUB(self, v):
        self._vR = self._ssat(self._double(self._VLD_check(v)) - self._double(self._vR))

    @staticmethod
    def __combine_helper(vh, vl, out_type, lowmask, bpe):
        return (out_type(vh) << bpe) + (out_type(vl) & lowmask)

    def _combine_singles(self, vh, vl):
        return self.__combine_helper(vh, vl, out_type=self._double,
                                             lowmask=self._lowmask,
                                             bpe=self._bpe)

    def _combine_doubles(self, vh, vl):
        if self._bpe == 32:
            raise ValueError("Combining double words is not allowed in 32-bit mode!")
        return self.__combine_helper(vh, vl, out_type=self._quad,
                                             lowmask=self._quadmask,
                                             bpe=self._quad(self._bpe*2))

    def _combine_vD_vR(self):
        if self._bpe == 8:
            vH = np.apply_along_axis(lambda p: self._combine_singles(p[1], p[0]), 
                                     arr=self._vD.reshape((-1, 2)), axis=1)
            vL = np.apply_along_axis(lambda p: self._combine_singles(p[1], p[0]), 
                                     arr=self._vR.reshape((-1, 2)), axis=1)
            return self.__combine_helper(vH, vL, out_type=self._quad,
                                         lowmask=self._vec_quadmask[::2],
                                         bpe=self._quad(self._bpe*2))
        else:
            return self.__combine_helper(self._vD, self._vR,
                                         out_type=self._double,
                                         lowmask=self._vec_lowmask,
                                         bpe=self._bpe)

    @staticmethod
    def __split_helper(v, out_type, lowmask, bpe):
        vlow, vhigh = out_type(v & lowmask), out_type((v >> bpe) & lowmask)
        return vlow, vhigh

    def _split_double(self, v):
        return self.__split_helper(v, out_type=self._single,
                                      lowmask=self._lowmask,
                                      bpe=self._bpe)

    def _split_quad(self, v):
        if self._bpe == 32:
            raise ValueError("Combining double words is not allowed in 32-bit mode!")
        return self.__split_helper(v, out_type=self._double,
                                      lowmask=self._quadmask,
                                      bpe=self._double(self._bpe*2))

    def _split_to_vD_vR(self, v):
        if self._bpe == 8:
            vL, vH = self.__split_helper(v, out_type=self._double,
                                            lowmask=self._vec_quadmask[::2],
                                            bpe=self._double(self._bpe*2))

            vLL, vLH = self.__split_helper(vL, out_type=self._single,
                                               lowmask=self._vec_lowmask[::2],
                                               bpe=self._bpe)
            self._vR[0::2], self._vR[1::2] = vLL, vLH

            vHL, vHH = self.__split_helper(vH, out_type=self._single,
                                               lowmask=self._vec_lowmask[::2],
                                               bpe=self._bpe)
            self._vD[0::2], self._vD[1::2] = vHL, vHH
        else:
            self._vR = self._single(np.bitwise_and(v, self._vec_lowmask))
            self._vD = self._single(np.bitwise_and(v >> self._bpe, self._vec_lowmask))

    def VLMACC(self, v):
        if self._bpe == 8:
            # TODO: change the subset here to lower half
            prod = self._quad(self._vC[::2]) * self._quad(self._VLD_check(v)[::2])
            self._split_to_vD_vR(prod + self._combine_vD_vR())
        elif self._bpe == 16:
            tmp = self._quad(self._vC) * self._quad(self._VLD_check(v))
            tmp = self._dsat(tmp + self._combine_vD_vR())
            self._split_to_vD_vR(tmp)
        else:
            raise NotImplementedError()

    def VLMACCR(self, v):
        if self._bpe == 8:
            vD_vR = self._combine_vD_vR()
            prod = self._quad(self._vC) * self._quad(self._VLD_check(v))
            vD_vR = np.hstack([vD_vR[-1] + np.sum(prod), vD_vR[:-1]])
            self._split_to_vD_vR(vD_vR)
        elif self._bpe == 16:
            vD_vR = self._combine_vD_vR()
            prod = self._quad(self._vC) * self._quad(self._VLD_check(v))
            tmp = self._dsat(self._quad(vD_vR[-1]) + np.sum(prod))
            vD_vR = np.hstack([tmp, vD_vR[:-1]])
            self._split_to_vD_vR(vD_vR)
        else:
            raise NotImplementedError()

