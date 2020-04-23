#ifndef MOBILENET_OPS_RESOLVER_H_
#define MOBILENET_OPS_RESOLVER_H_

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

void add_registered_ops(MicroMutableOpResolver *resolver);

class MobileNetOpsResolver : public MicroMutableOpResolver {
 public:
  MobileNetOpsResolver();

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // MOBILENET_OPS_RESOLVER_H_
