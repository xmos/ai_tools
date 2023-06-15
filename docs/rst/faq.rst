Frequently Asked Questions
==========================


Pytorch conversion produces unexpected results
----------------------------------------------

You can set an ONNX version number to use for the conversion. You may want
to experiment with different values, because some modern operators are
mapped to a sequence of basic operators rather than a single modern
operators if an old version of ONNX is used.
