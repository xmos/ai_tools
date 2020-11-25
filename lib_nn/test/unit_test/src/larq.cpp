

#include "larq_compute_engine/core/bitpacking/bitpack.h"
#include "larq_compute_engine/core/bconv2d/output_transform.h"
#include "larq_compute_engine/core/types.h"
#include "nn_operator.h"

using namespace tflite;

#include "larq_compute_engine/core/bconv2d/reference.h"

namespace compute_engine {
  
namespace ce = compute_engine;

namespace core {

extern "C" void larq_ref_bsign(int8_t* input, int32_t* output,
                               size_t inputLength, int32_t zero_point) {
  bitpacking::bitpack_array<std::int8_t>(
      input, inputLength, output, zero_point);
}

using bconv2d::OutputTransform;

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

template <typename DstScalar>
void conv2d_larq_impl(const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_params_t* k,
    const int32_t* packed_input_data,
    const int32_t* packed_filter_data,
    DstScalar* packed_output_data,
    const unsigned channels_per_output_word,
    const ce::core::OutputTransform<DstScalar> &output_transform
  ){

    int x_dims[4];
    int k_dims[4];
    int y_dims[4];

    const int batches = 1;
    const int channels_per_word = 32;

    x_dims[0] = batches;
    x_dims[1] = x->height;
    x_dims[2] = x->width;
    x_dims[3] = x->channels / channels_per_word;

    k_dims[0] = y->channels;
    k_dims[1] = k->shape.height;
    k_dims[2] = k->shape.width;
    k_dims[3] = x->channels / channels_per_word;

    y_dims[0] = batches;
    y_dims[1] = y->height;
    y_dims[2] = y->width;
    y_dims[3] = y->channels / channels_per_output_word;
    
    compute_engine::core::bconv2d::BConv2DParams params;

    params.filter_width = k->shape.height;
    params.filter_height = k->shape.width;
    params.channels_in = x->channels / channels_per_word;
    params.channels_out = y->channels / channels_per_output_word;
    params.groups = 1;

    params.dilation_height_factor = k->dilation.vertical;
    params.dilation_width_factor = k->dilation.horizontal;
    params.padding_type = TfLitePadding::kTfLitePaddingValid; 

    params.padding_values.height = 0;
    params.padding_values.height_offset = 0;
    params.padding_values.width = 0;
    params.padding_values.width_offset = 0;

    params.stride_height = k->stride.vertical;
    params.stride_width = k->stride.horizontal;

    RuntimeShape packed_input_shape  = RuntimeShape(4, (const int32_t*)x_dims);
    RuntimeShape output_shape        = RuntimeShape(4, (const int32_t*)y_dims);
    RuntimeShape packed_filter_shape = RuntimeShape(4, (const int32_t*)k_dims);


    compute_engine::core::bconv2d::BConv2DReference<std::uint32_t, DstScalar>(
      &params, packed_input_shape, packed_input_data,
      packed_filter_shape, packed_filter_data,
      output_transform, output_shape, packed_output_data);
}

extern "C" void larq_ref_bconv2d_int8_out(const nn_image_params_t* x,
                                 const nn_image_params_t* y,
                                 const nn_window_params_t* k,
                                 const int32_t* packed_input_data,
                                 const int32_t* packed_filter_data,
                                 int8_t* packed_output_data,
                                 const float* post_activation_multiplier, 
                                 const float* post_activation_bias 
) {
  ce::core::OutputTransform<std::int8_t> output_transform;
  ce::core::GetOutputTransform(output_transform, 0, INT32_MAX, 
    post_activation_multiplier, post_activation_bias);

  const unsigned channels_per_output_word = 1;

  conv2d_larq_impl <std::int8_t> (x, y, k, packed_input_data, packed_filter_data, packed_output_data, 
    channels_per_output_word, output_transform);
}

extern "C" void larq_ref_bconv2d_bin_out(const nn_image_params_t* x,
                                 const nn_image_params_t* y,
                                 const nn_window_params_t* k,
                                 const int32_t* packed_input_data,
                                 const int32_t* packed_filter_data,
                                 int32_t* packed_output_data,
                                 const int32_t* thresholds
) {

  ce::core::OutputTransform<std::int32_t> output_transform;
  ce::core::GetOutputTransform(output_transform, thresholds);
  
  const unsigned channels_per_output_word = 32;
  
  conv2d_larq_impl <std::int32_t> (x, y, k, packed_input_data, packed_filter_data, packed_output_data, 
    channels_per_output_word, output_transform);

}

//}  // namespace ref
}  // namespace compute_engine
