// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"
include "Utils/Utils.td"

// Convert the StridedSliceOp to a SliceOp when possible. This will enable other
// optimizations on SliceOp to run.
def OptimizeStridedSlice : Pat<
  (TFL_StridedSliceOp $input,
    (Arith_ConstantOp $begin),
    (Arith_ConstantOp $end),
    (Arith_ConstantOp $stride),
    ZeroIntAttr:$_,
    ZeroIntAttr:$_,
    ZeroIntAttr:$_,
    ZeroIntAttr:$_,
    ZeroIntAttr:$_,
    ConstBoolAttrFalse),
  (TFL_SliceOp $input,
    (Arith_ConstantOp $begin),
    (Arith_ConstantOp (GetOffSet $begin, $end))),
  [(IsAllOnesConstant $stride)]>;


// Convert Quantize(Reshape()) -> Reshape(Quantize())
// This is to merge Quantize with Conv2D if possible
def : Pat<(TFL_QuantizeOp(TFL_ReshapeOp $input, $shape), $qtype),
          (TFL_ReshapeOp(TFL_QuantizeOp $input,
                         (UpdateShapeWithAxis<-1> $qtype, $input)),
           $shape),
          [(IsQuantizedType $input)]>;

// Fuse Quantize(Conv2D()) -> Conv2D()
def : Pat<(TFL_QuantizeOp(TFL_Conv2DOp $input, $f, $b, $dh, $dw, $faf, $p, $sh,
                          $sw),
           $qtype),
          (TFL_Conv2DOp $input, $f, $b, $dh, $dw, $faf, $p, $sh, $sw)>;

def : Pat<(TFL_QuantizeOp(TFL_DepthwiseConv2DOp $input, $f, $b, $dh, $dw, $faf,
                          $p, $sh, $sw, $dm),
           $qtype),
          (TFL_DepthwiseConv2DOp $input, $f, $b, $dh, $dw, $faf, $p, $sh, $sw,
           $dm)>;

// TFL_Conv2D() with SAME padding -> TFL_Conv2D(Pad())
def : Pat<(TFL_Conv2DOp: $output TensorOf<[QI8, QI16]>:$input, TensorOf<[QI8]>:$f, AnyTypeOf<[TensorOf<[I32,QI32]>, NoneType]>:$b, $dh, $dw, $faf, TFL_PAD_Same, $sh, $sw),
          (TFL_Conv2DOp (TFL_PadOp $input,
                        (GetConv2DPaddingValues
                         : $ret__0 $output),
                        (returnType $ret__1)), $f, $b, $dh, $dw, $faf, TFL_PAD_Valid, $sh, $sw)>;

// TFL_DepthwiseConv2D() with SAME padding-> TFL_DepthwiseConv2D(Pad())
def : Pat<(TFL_DepthwiseConv2DOp: $output TensorOf<[QI8]>:$input, TensorOf<[QI8]>:$f, TensorOf<[I32,QI32]>:$b, $dh, $dw, $faf, TFL_PAD_Same, $sh, $sw, $dm),
          (TFL_DepthwiseConv2DOp (TFL_PadOp $input,
                        (GetDepthwiseConv2DPaddingValues
                         : $ret__0 $output),
                        (returnType $ret__1)), $f, $b, $dh, $dw, $faf, TFL_PAD_Valid, $sh, $sw, $dm)>;

def : Pat<(TFL_ReluOp(TFL_MinimumOp $lhs, $rhs)), (TFL_ReluOp $lhs),
          [(IsSplatAndEqualTo<127> $rhs)]>;

// Merge Relu with Conv
def : Pat<(TFL_ReluOp(TFL_Conv2DOp $input, $f, $b, $dh, $dw, TFL_AF_None, $p,
                      $sh, $sw)),
          (TFL_Conv2DOp $input, $f, $b, $dh, $dw, TFL_AF_Relu, $p, $sh, $sw)>;

// If MeanOp with spatial axis and rank 2 output, expand output to rank 4, which
// we later lower to AveragePool2D
def : Pat<(TFL_MeanOp
           : $output $input,
             (TFL_ConstOp
              : $axis_op $axis),
             $kd),
          (TFL_ReshapeOp(TFL_MeanOp $input, $axis_op, $kd,
                         (returnType(getExpandedShape $output))),
           (TFL_ConstOp(getExpandedShapeAttr $output))),
          [(HasSpatialAxisForMean $axis), (HasRank<2> $output)]>;

// Lower MeanOp with spatial axis to AveragePool2D
def : Pat<(TFL_MeanOp
           : $output $input, (TFL_ConstOp $axis), $kd),
          (TFL_QuantizeOp(
               TFL_AveragePool2DOp $input, (GetDimAsI32<1> $input),
               (GetDimAsI32<2> $input), TFL_PAD_Valid,
               ConstantAttr<I32Attr, "1">, ConstantAttr<I32Attr, "1">,
               TFL_AF_None,
               (returnType(getTypeOf1WithQParamsOf0 $input, $output))),
           (getTypeAttrOf1WithQParamsOf0 $output, $output)),
          [(HasSpatialAxisForMean $axis), (HasRank<4> $output)]>;

// PadChannel(PadSpatial) to PadSpatial(PadChannel)
// Match cases where arith constant op and tfl constant op are both used
foreach constOp = [Arith_ConstantOp, TFL_ConstOp] in {
def:
  Pat<(TFL_PadOp
       : $output(TFL_PadOp $input,
                 (constOp
                  : $padding_spatial_op $padding_spatial_attr)),
         (constOp
          : $padding_channel_op $padding_channel_attr)),
      (TFL_PadOp(TFL_PadOp $input, $padding_channel_op,
                 (returnType(PadChannelOutputType $input, $output))),
       $padding_spatial_op),
      [
        (HasOnlySpatialPadding $padding_spatial_attr),
        (HasOnlyChannelPadding $padding_channel_attr),
      ]>;
}

def : Pat<(TFL_PadOp
           : $output(TFL_PadOp $input,
                     (Arith_ConstantOp
                      : $padding_spatial_op $padding_spatial_attr)),
             (TFL_ConstOp
              : $padding_channel_op $padding_channel_attr)),
          (TFL_PadOp(TFL_PadOp $input, $padding_channel_op,
                     (returnType(PadChannelOutputType $input, $output))),
           $padding_spatial_op),
          [
            (HasOnlySpatialPadding $padding_spatial_attr),
            (HasOnlyChannelPadding $padding_channel_attr),
          ]>;

def : Pat<(TFL_PadOp
           : $output(TFL_PadOp $input,
                     (TFL_ConstOp
                      : $padding_spatial_op $padding_spatial_attr)),
             (Arith_ConstantOp
              : $padding_channel_op $padding_channel_attr)),
          (TFL_PadOp(TFL_PadOp $input, $padding_channel_op,
                     (returnType(PadChannelOutputType $input, $output))),
           $padding_spatial_op),
          [
            (HasOnlySpatialPadding $padding_spatial_attr),
            (HasOnlyChannelPadding $padding_channel_attr),
          ]>;

// Replace LQ_Bconv2DOp of SAME padding with a pad_value of one with
// LQ_Bconv2DOp(TFL_Pad()) of VALID padding. We cannot do this when the
// pad_value is zero as detailed below.
// Comment copied from
// https://github.com/larq/compute-engine/blob/main/larq_compute_engine/core/bconv2d/zero_padding_correction.h#L6
// "When we compute a convolution that requires "SAME" padding we pad with the
// value zero, meaning bitpacked bits 0, representing the value +1; thus we
// compute 'same one' padding by default. A correction is needed if we want
// 'same zero' padding instead -- we have to add or subtract a value to elements
// at the edge of the output tensor."
def : Pat<(LQ_Bconv2dOp
           : $output $input, $f, $m, $b, $t, $cin, $dh, $dw, $faf,
             ConstantAttr<I32Attr, "1">, TFL_PAD_Same, $sh, $sw),
          (LQ_Bconv2dOp(TFL_PadOp $input,
                        (GetBConv2DPaddingValues
                         : $ret__0 $output),
                        (returnType $ret__1)),
           $f, $m, $b, $t, $cin, $dh, $dw, $faf, ConstantAttr<I32Attr, "0">,
           TFL_PAD_Valid, $sh, $sw)>;
