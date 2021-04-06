// This is the optimization pattern definition file for XCore.

include "../ir/xc_ops.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"

// Checks if the value has only one user.
def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

/*

// Checks if the operand0's rank is one less than operand1's rank.
def PReluAlphaRankCheck : Constraint<
  CPred<"$0.getType().cast<ShapedType>().getRank() == "
  "$1.getType().cast<ShapedType>().getRank() - 1">>;

// PReLU pattern from Keras:
// f(x) = Relu(x) + (-alpha * Relu(-x))
def : Pat<(TF_AddV2Op
           (TF_ReluOp:$relu_out $x),
           (TF_MulOp:$mul_out
            (TF_ReluOp (TF_NegOp:$input_neg_out $x)),
            $neg_alpha)),
          (TFJS_PReluOp $x, (TF_NegOp $neg_alpha)),
          [(PReluAlphaRankCheck $neg_alpha, $x),
           (HasOneUse $relu_out),
           (HasOneUse $mul_out),
           (HasOneUse $input_neg_out)
          ]>;
*/
