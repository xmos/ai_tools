// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

// This is the optimization pattern definition file for XCore.
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"

include "IR/XCoreOps.td"

// Activation lowering patterns
def getLookupTable
    : NativeCodeCall<"getLookupTable($_builder, $0.getDefiningOp())">;

// TODO: DPK
// foreach activationOp =
//     [TFL_ReluOp, TFL_Relu6Op, TFL_TanhOp, TFL_LogisticOp] in {
// def:
//   Pat<(activationOp
//             : $output TensorOf<[QI8]>:$input),
//             (XC_Lookup8Op $input, (ConstantOp (getLookupTable $output)))>;
// }

// Pad pattern
// We only support padding for 4 dimensions
def HasValidPaddingShape
    : Constraint<CPred<"$0.getType().cast<ShapedType>().getDimSize(0) == 4">>;

// We only support spatial padding
def HasNoPaddingForBatchAndChannel
    : Constraint<CPred<
          "$0.cast<DenseIntElementsAttr>().getNumElements() == 8 && "
          "($0.cast<DenseIntElementsAttr>().getValue<int32_t>({0,0}) == 0 &&"
          "$0.cast<DenseIntElementsAttr>().getValue<int32_t>({0,1}) == 0 &&"
          "$0.cast<DenseIntElementsAttr>().getValue<int32_t>({3,0}) == 0 &&"
          "$0.cast<DenseIntElementsAttr>().getValue<int32_t>({3,1}) == 0)">>;

// We only support bytes per pixel being a multiple of four
// Check if channel dimension size * sizeof(input type) is divisible by 4
def HasValidBytesPerPixel : Constraint<Or<[
  CPred<"$0.getType().cast<ShapedType>().getElementType().isIntOrFloat() &&"
        "($0.getType().cast<ShapedType>().getElementType()."
        "getIntOrFloatBitWidth() / 8 *  "
        "$0.getType().cast<ShapedType>().getDimSize(3)) % 4 == 0">,
  CPred<"$0.getType().cast<ShapedType>().getElementType().isa<quant::"
        "QuantizedType>() &&"
        "($0.getType().cast<ShapedType>().getElementType().cast<quant::"
        "QuantizedType>().getStorageTypeIntegralWidth() / 8 * "
        "$0.getType().cast<ShapedType>().getDimSize(3)) % 4 == 0">
]>>;

def getPadValue : NativeCodeCall<"getPadValue($_builder, $0)">;

// TODO: Remove when we have fusing of Pad and Conv2D in xformer2
// Replacing pad when there is a following Conv2D after a Pad is breaking fusing
// passes in xformer1
def HasNoFollowingConv2D : Constraint<CPred<"hasNoFollowingConv2D($0)">>;

// TODO : DPK
// Not using XC_Pad in our runtime at the moment
// Would fail with "Failed to get registration from op code CUSTOM"
// def : Pat<(TFL_PadOp
//            : $output $input, (TFL_ConstOp
//                               : $padding_op $padding_attr)),
//           (XC_PadOp $input, $padding_op, (getPadValue $input)), [
//             (HasValidPaddingShape $padding_op),
//             (HasNoPaddingForBatchAndChannel $padding_attr),
//             (HasValidBytesPerPixel $input), (HasNoFollowingConv2D $output)
//           ]>;

// Fuse XC_Conv2D(Reshape()) -> XC_Conv2D()
def : Pat<(XC_Conv2DV2Op
           : $cout(TFL_ReshapeOp
                   : $rout $input, $shape),
             $weights, $muls, $kt, $mp, $aggp, $otp, $scratch, $tc, $akp),
          (XC_Conv2DV2Op $input, $weights, $muls, $kt, $mp, $aggp, $otp,
           $scratch, $tc, $akp)>;

// Fuse Reshape(XC_Conv2D()) -> XC_Conv2D()
def : Pat<(TFL_ReshapeOp
           : $rout(XC_Conv2DV2Op $input, $weights, $muls, $kt, $mp, $aggp, $otp,
                   $scratch, $tc, $akp),
             $shape),
          (XC_Conv2DV2Op $input, $weights, $muls, $kt, $mp, $aggp, $otp,
           $scratch, $tc, $akp)>;

// Replace LQ_QuantizeOp with XC_bsign_8
def : Pat<(LQ_QuantizeOp $input), (XC_Bsign8Op $input)>;

def getMemcpyFnParam : NativeCodeCall<"getMemcpyFnParam($0)">;

def getI32IntegerAttrZero : NativeCodeCall<"getI32IntegerAttrZero">;

def HasNoFollowingStridedSlice : Constraint<CPred<"HasNoFollowingStridedSlice($0)">>;
// First pass of op split
// Insert StidedSliceOp
def : Pat<( TFL_Conv2DOp:$conv_out $input, $filter, $bias,
             $h_factor, $w_factor, TFL_AF_None, $padding,
              $stride_h, $stride_w),
            (XC_StridedSliceOp (TFL_Conv2DOp $input, $filter,
             $bias, $h_factor, $w_factor, TFL_AF_None, $padding,
              $stride_h, $stride_w), (getI32IntegerAttrZero),
             (getI32IntegerAttrZero), $h_factor), 
            [(HasNoFollowingStridedSlice $conv_out)]>;
