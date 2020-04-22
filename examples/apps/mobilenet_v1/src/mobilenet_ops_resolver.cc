#include "mobilenet_ops_resolver.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

void add_registered_ops(MicroMutableOpResolver *resolver) {
  resolver->AddBuiltin(BuiltinOperator_SOFTMAX, Register_SOFTMAX(), 1, 2);
  resolver->AddBuiltin(BuiltinOperator_PAD, Register_PAD(), 1, 2);
  resolver->AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D(), 1, 3);
  resolver->AddCustom("XC_conv2d_depthwise", Register_Conv2D_depthwise());
  resolver->AddCustom("XC_conv2d_1x1", Register_Conv2D_1x1());
  resolver->AddCustom("XC_avgpool2d_global", Register_AvgPool2D_Global());
  resolver->AddCustom("XC_fc_deepin_anyout", Register_FullyConnected_16());
  resolver->AddCustom("XC_requantize_16_to_8", Register_Requantize_16_to_8());
}

MobileNetOpsResolver::MobileNetOpsResolver() { add_registered_ops(this); }

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite