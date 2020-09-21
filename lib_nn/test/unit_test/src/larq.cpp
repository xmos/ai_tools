#include "larq_compute_engine/core/bconv2d_impl_ref.h"
#include "larq_compute_engine/core/bitpack.h"

#include "larq_compute_engine/core/bconv2d_output_transform.h"
#include "larq_compute_engine/core/types.h"
using namespace tflite;

namespace compute_engine {
namespace ce = compute_engine;

namespace core {

extern "C" void larq_ref_bsign(int8_t* input, int32_t* output,
                               size_t inputLength, int32_t zero_point) {
  bitpack_array<std::int8_t>(
      input, inputLength, output, zero_point);
}

using compute_engine::core::OutputTransform;

// Fill in the OutputTransform values for float and/or int8 outputs
template <typename DstScalar>
void GetOutputTransform(OutputTransform<DstScalar>& output_transform,
                        int32_t output_transform_clamp_min,
                        int32_t output_transform_clamp_max,
                        const float * output_transform_multiplier,
                        const float * output_transform_bias) {
  static_assert( std::is_same<DstScalar, std::int8_t>::value, "");
  output_transform.clamp_min = output_transform_clamp_min;
  output_transform.clamp_max = output_transform_clamp_max;
  output_transform.multiplier = output_transform_multiplier;
  output_transform.bias = output_transform_bias;
}

// Fill in the OutputTransform values for bitpacked outputs
void GetOutputTransform(OutputTransform<core::TBitpacked>& output_transform,
                        const int32_t* thresholds) {
  output_transform.thresholds = thresholds;
}
}

#include "nn_operator.h"
#include "nn_op_structs.h"

// Fill the OutputTransform values for bitpacked int32 outputs


extern "C" void larq_ref_bconv2d_int8_out(const nn_image_params_t* x,
                                 const nn_image_params_t* y,
                                 const nn_window_params_t* k,
                                 const int32_t* packed_input_data,
                                 const int32_t* packed_filter_data,
                                 int8_t* output_data,
                                 const float* post_activation_multiplier, 
                                 const float* post_activation_bias 

) {
  ConvParams params;

  const int batches = 1;
  int x_dims[4] = {batches, (int)x->height, (int)x->width,
                   (int)x->channels / 32};
  int k_dims[4] = {(int)y->channels, (int)k->shape.height, (int)k->shape.width,
                   (int)x->channels / 32};
  int y_dims[4] = {batches, (int)y->height, (int)y->width,
                   (int)y->channels};

  RuntimeShape packed_input_shape = RuntimeShape(4, (const int32*)x_dims);
  RuntimeShape packed_filter_shape = RuntimeShape(4, (const int32*)k_dims);
  RuntimeShape output_shape = RuntimeShape(4, (const int32*)y_dims);

  //   OutputTransform<AccumScalar, DstScalar>
  ce::core::OutputTransform<std::int8_t> output_transform;
  ce::core::GetOutputTransform(output_transform, 
    0, INT32_MAX, 
    post_activation_multiplier, 
    post_activation_bias);

  params.dilation_height_factor = k->dilation.vertical;
  params.dilation_width_factor = k->dilation.horizontal;
  //   params.input_offset = 0;
  //   params.output_multiplier = 0;
  //   params.output_offset = 0;
  //   params.output_shift = 0;
  params.padding_type = PaddingType::kValid;  // enum class PaddingType : uint8
                                              // { kNone, kSame, kValid };
  params.padding_values.height = 0;
  params.padding_values.height_offset = 0;
  params.padding_values.width = 0;
  params.padding_values.width_offset = 0;
  //   params.quantized_activation_max = 0;
  //   params.quantized_activation_min = 0;
  params.stride_height = k->stride.vertical;
  params.stride_width = k->stride.horizontal;
  //   params.weights_offset = 0;

  RuntimeShape no_shape = RuntimeShape(0, nullptr);
  ce::ref::BConv2D<std::int32_t, std::int8_t, core::OutputTransformDetails::Default>(
      params, packed_input_shape, packed_input_data,
      packed_filter_shape, packed_filter_data,
      output_transform, output_shape, output_data,

      // These are all dummy parameters(unused)
      no_shape, 0, 0, 0, 0);
}

extern "C" void larq_ref_bconv2d_bin_out(const nn_image_params_t* x,
                                 const nn_image_params_t* y,
                                 const nn_window_params_t* k,
                                 const int32_t* packed_input_data,
                                 const int32_t* packed_filter_data,
                                 int32_t* packed_output_data,
                                 const long* thresholds

) {
  ConvParams params;

  const int batches = 1;
  int x_dims[4] = {batches, (int)x->height, (int)x->width,
                   (int)x->channels / 32};
  int k_dims[4] = {(int)y->channels, (int)k->shape.height, (int)k->shape.width,
                   (int)x->channels / 32};
  int y_dims[4] = {batches, (int)y->height, (int)y->width,
                   (int)y->channels / 32};

  RuntimeShape packed_input_shape = RuntimeShape(4, (const int32*)x_dims);
  RuntimeShape packed_filter_shape = RuntimeShape(4, (const int32*)k_dims);
  RuntimeShape output_shape = RuntimeShape(4, (const int32*)y_dims);

  //   OutputTransform<AccumScalar, DstScalar>
  ce::core::OutputTransform<std::int32_t> output_transform;
  ce::core::GetOutputTransform(output_transform, (const int32_t*)thresholds);

  params.dilation_height_factor = k->dilation.vertical;
  params.dilation_width_factor = k->dilation.horizontal;
  //   params.input_offset = 0;
  //   params.output_multiplier = 0;
  //   params.output_offset = 0;
  //   params.output_shift = 0;
  params.padding_type = PaddingType::kValid;  // enum class PaddingType : uint8
                                              // { kNone, kSame, kValid };
  params.padding_values.height = 0;
  params.padding_values.height_offset = 0;
  params.padding_values.width = 0;
  params.padding_values.width_offset = 0;
  //   params.quantized_activation_max = 0;
  //   params.quantized_activation_min = 0;
  params.stride_height = k->stride.vertical;
  params.stride_width = k->stride.horizontal;
  //   params.weights_offset = 0;

  RuntimeShape no_shape = RuntimeShape(0, nullptr);
  ce::ref::BConv2D<std::uint32_t, std::int32_t, core::OutputTransformDetails::Default>(
      params, packed_input_shape, packed_input_data,
      packed_filter_shape, packed_filter_data,
      output_transform, output_shape, packed_output_data,

      // These are all dummy parameters(unused)
      no_shape, 0, 0, 0, 0);
}
//}  // namespace ref
}  // namespace compute_engine
