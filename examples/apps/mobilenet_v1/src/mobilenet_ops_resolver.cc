#include "mobilenet_ops_resolver.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

void add_registered_ops(MicroMutableOpResolver<num_mobilenet_ops> *resolver) {
  resolver->AddSoftmax();
  resolver->AddPad();
  resolver->AddCustom(Conv2D_Shallow_OpCode, Register_Conv2D_Shallow());
  resolver->AddCustom(Conv2D_Depthwise_OpCode, Register_Conv2D_Depthwise());
  resolver->AddCustom(Conv2D_1x1_OpCode, Register_Conv2D_1x1());
  resolver->AddCustom(AvgPool2D_Global_OpCode, Register_AvgPool2D_Global());
  resolver->AddCustom("XC_fc_deepin_anyout", Register_FullyConnected_16());
  resolver->AddCustom(Requantize_16_to_8_OpCode, Register_Requantize_16_to_8());
}

MobileNetOpsResolver::MobileNetOpsResolver() { add_registered_ops(this); }

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite