#ifndef MOBILENET_OPS_RESOLVER_H_
#define MOBILENET_OPS_RESOLVER_H_

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

constexpr int num_mobilenet_ops = 8;

void add_registered_ops(MicroMutableOpResolver<num_mobilenet_ops> *resolver);

class MobileNetOpsResolver : public MicroMutableOpResolver<num_mobilenet_ops> {
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
