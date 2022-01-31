# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class SVDFOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SVDFOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsSVDFOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def SVDFOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # SVDFOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SVDFOptions
    def Rank(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # SVDFOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def SVDFOptionsStart(builder): builder.StartObject(2)
def Start(builder):
    return SVDFOptionsStart(builder)
def SVDFOptionsAddRank(builder, rank): builder.PrependInt32Slot(0, rank, 0)
def AddRank(builder, rank):
    return SVDFOptionsAddRank(builder, rank)
def SVDFOptionsAddFusedActivationFunction(builder, fusedActivationFunction): builder.PrependInt8Slot(1, fusedActivationFunction, 0)
def AddFusedActivationFunction(builder, fusedActivationFunction):
    return SVDFOptionsAddFusedActivationFunction(builder, fusedActivationFunction)
def SVDFOptionsEnd(builder): return builder.EndObject()
def End(builder):
    return SVDFOptionsEnd(builder)