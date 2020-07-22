#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>
#include <stdio.h>
#include <array>

#include "larq_compute_engine/core/bconv2d_output_transform.h"
#include "larq_compute_engine/core/bconv2d_impl_ref.h"
#include "larq_compute_engine/core/bgemm_functor.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/internal/common.h"

using namespace tflite;

namespace compute_engine {
namespace ce = compute_engine;
namespace ref {

template <typename AccumScalar, typename DstScalar>
struct OutputTransform {};

// DstScalar = int32_t

// A part of the transformation is common to all output types.
// This part is described by `OutputTransformBase`.
template <typename AccumScalar>
struct OutputTransformBase {
  AccumScalar backtransform_add = 0;
  std::int32_t clamp_min = std::numeric_limits<AccumScalar>::lowest();
  std::int32_t clamp_max = std::numeric_limits<AccumScalar>::max();

  inline AccumScalar RunBase(const AccumScalar accum) const {
    // Backtransform can still be done in int32
    AccumScalar x = backtransform_add - 2 * accum;
    // Activation function can also be done in int32
    x = std::min<AccumScalar>(x, clamp_max);
    x = std::max<AccumScalar>(x, clamp_min);
    return x;
  }
};

// Output transformation for bitpacked output
template <typename AccumScalar>
struct OutputTransform<AccumScalar, std::int32_t> {
  const AccumScalar* thresholds = nullptr;

  bool Run(const AccumScalar accum, int out_channel) const {
    TF_LITE_ASSERT(thresholds != nullptr);
    return accum > thresholds[out_channel];
  }
};

// Output transformation for 8-bit quantization

template <typename AccumScalar>
struct OutputTransform<AccumScalar, std::int8_t>
    : OutputTransformBase<AccumScalar> {
  // These effective values are the post-activation multipliers and biases
  // divided by output_scale and including the output zero_point
  const float* effective_post_activation_multiplier = nullptr;
  const float* effective_post_activation_bias = nullptr;

  std::int8_t Run(const AccumScalar accum, int out_channel) const {
    // First convert to full precision to do the linear transformation
    float result_fp = static_cast<float>(this->RunBase(accum));
    result_fp *= effective_post_activation_multiplier[out_channel];
    result_fp += effective_post_activation_bias[out_channel];
    // Now round back to int32
    AccumScalar result = tflite::TfLiteRound(result_fp);
    // Clamp to int8 range
    result =
        std::min<std::int32_t>(result, std::numeric_limits<std::int8_t>::max());
    result = std::max<std::int32_t>(result,
                                    std::numeric_limits<std::int8_t>::lowest());
    return static_cast<std::int8_t>(result);
  }
};

// Fill the OutputTransform values for bitpacked int32 outputs
template <typename AccumScalar>
void GetOutputTransform(
    const long* thresholds,
    OutputTransform<AccumScalar, std::int32_t>& output_transform) {
  output_transform.thresholds = thresholds;
}

template <typename TBitpacked, typename AccumScalar, typename DstScalar>
inline void BConv2D(
    const ConvParams& params, const RuntimeShape& packed_input_shape,
    const TBitpacked* packed_input_data,
    const RuntimeShape& packed_filter_shape,
    const TBitpacked* packed_filter_data,
    const OutputTransform<std::int32_t, DstScalar>& output_transform,
    const RuntimeShape& output_shape, DstScalar* output_data) {
  static_assert(std::is_same<DstScalar, float>::value ||
                    std::is_same<DstScalar, std::int32_t>::value ||
                    std::is_same<DstScalar, std::int8_t>::value,
                "The reference implementation supports either float "
                "output, 32-bit bitpacked output or 8-bit quantized output.");

  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  TFLITE_DCHECK_EQ(packed_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(packed_filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  //   (void)im2col_data;   // only used in optimized code.
  //   (void)im2col_shape;  // only used in optimized code.
  const int batches = MatchingDim(packed_input_shape, 0, output_shape, 0);
  const int input_depth =
      MatchingDim(packed_input_shape, 3, packed_filter_shape, 3);
  const int output_depth = packed_filter_shape.Dims(0);
  const int input_height = packed_input_shape.Dims(1);
  const int input_width = packed_input_shape.Dims(2);
  const int filter_height = packed_filter_shape.Dims(1);
  const int filter_width = packed_filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        // This variable is only used if we are writing bitpacked output.
        std::uint32_t bitpacked_column = 0;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          AccumScalar accum = AccumScalar(0);
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // `pad_value=1`, which means the bitpacked value is 0, so we
                // set `input_value=0`
                TBitpacked input_value = 0;
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  input_value = packed_input_data[Offset(
                      packed_input_shape, batch, in_y, in_x, in_channel)];
                }
                TBitpacked filter_value =
                    packed_filter_data[Offset(packed_filter_shape, out_channel,
                                              filter_y, filter_x, in_channel)];
                accum += ce::core::xor_popcount<TBitpacked, AccumScalar>(
                    input_value, filter_value);
              }
            }
          }
          // If the destination scalar is int32, we're writing bitpacked output.
          if (std::is_same<DstScalar, std::int32_t>::value) {
            // printf("l %u %u %u %d\n", out_y, out_x, out_channel, accum);
            bool bit = output_transform.Run(accum, out_channel);
            if (bit) bitpacked_column |= 1ULL << (out_channel % 32);

            // After we've 'filled' the `bitpacked_column` with 32 values, or
            // reached the end of the channels, we write it to memory.
            if ((out_channel + 1) % 32 == 0 ||
                (out_channel + 1 == output_depth)) {
              output_data[Offset(output_shape, batch, out_y, out_x,
                                 out_channel / 32)] = bitpacked_column;
              // printf("%08x\n", bitpacked_column);
              bitpacked_column = 0;
            }
          }
          // Otherwise, we're not writing bitpacked output; it must be int8 or
          // float.
          else {
            DstScalar dst_val = output_transform.Run(accum, out_channel);
            output_data[Offset(output_shape, batch, out_y, out_x,
                               out_channel)] = dst_val;
          }
        }
      }
    }
  }
}

#include "nn_operator.h"
#include "nn_op_helper.h"
#include "nn_op_structs.h"
#include "xs3_vpu.h"

// Fill the OutputTransform values for bitpacked int32 outputs

extern "C" void larq_ref_bconv2d(const nn_image_params_t* x,
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
  OutputTransform<std::int32_t, std::int32_t> output_transform;
  GetOutputTransform(thresholds, output_transform);

  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  //   params.input_offset = 0;
  //   params.output_multiplier = 0;
  //   params.output_offset = 0;
  //   params.output_shift = 0;
  params.padding_type = PaddingType::kValid;  // enum class PaddingType : uint8
                                              // { kNone, kSame, kValid };
  params.padding_values.height = 0;
  //   params.padding_values.height_offset = 0;
  params.padding_values.width = 0;
  //   params.padding_values.width_offset = 0;
  //   params.quantized_activation_max = 0;
  //   params.quantized_activation_min = 0;
  params.stride_height = 1;
  params.stride_width = 1;
  //   params.weights_offset = 0;

  BConv2D<std::uint32_t, std::int32_t, std::int32_t>(
      params, packed_input_shape, (const uint32_t*)packed_input_data,
      packed_filter_shape, (const uint32_t*)packed_filter_data,
      output_transform, output_shape, packed_output_data);
}

//////////////////////
// bsign
//
enum class BitpackOrder { Canonical, Optimized };

template <class TIn, class TOut>
inline void pack_canonical(const TIn* fptr, TOut* buf) {
  constexpr std::size_t bitwidth = std::numeric_limits<TOut>::digits;
  *buf = 0;
  for (size_t i = 0; i < bitwidth; ++i) {
    if (fptr[i] < 0) *buf |= (TOut(1) << i);
  }
}

template <class TIn, class TOut>
inline void pack_canonical_quantized(const TIn* in, TOut* out,
                                     const std::int32_t zero_point) {
  constexpr std::size_t bitwidth = std::numeric_limits<TOut>::digits;
  *out = 0;
  for (size_t i = 0; i < bitwidth; ++i) {
    // Note: uint8 to int32 will set the top 24 bits to 0
    //        int8 to int32 will set the top 24 bits to the int8 sign bit
    if (static_cast<std::int32_t>(in[i]) < zero_point) *out |= (TOut(1) << i);
  }
}

// Helper function
template <BitpackOrder bitpack_order, class TIn, class TOut>
inline void pack_bitfield(const TIn* in, TOut* out,
                          const std::int32_t zero_point) {
  // Note: The expressions in these if-statements are known at compile-time so
  // they are all optimied away
  constexpr bool is_quantized = !std::is_floating_point<TIn>::value;
  if (bitpack_order == BitpackOrder::Canonical) {
    if (is_quantized)
      pack_canonical_quantized(in, out, zero_point);
    else
      pack_canonical(in, out);
  } 
  /*
   else {
    if (is_quantized)
      pack_optimized_quantized(in, out, zero_point);
    else
      pack_optimized(in, out);
  }
  */
}

template <BitpackOrder bitpack_order, class TIn, class TOut>
inline void packbits_array(const TIn* input_array, const std::size_t n,
                           TOut* bitpacked_array,
                           const std::int32_t zero_point) {
  constexpr std::size_t bitwidth = std::numeric_limits<TOut>::digits;

  int num_packed_elems = n / bitwidth;
  int elements_left = n - bitwidth * num_packed_elems;

  const TIn* in = input_array;
  TOut* out = bitpacked_array;

  while (num_packed_elems--) {
    pack_bitfield<bitpack_order>(in, out++, zero_point);
    in += bitwidth;
  }

  // If padding is needed, copy the remaining elements to a buffer and add
  // enough zeros to fill the bitwidth. This function assumes enough memory for
  // padding is already allocatd in the output array `bitpacked_array`.
  if (elements_left != 0) {
    std::array<TIn, bitwidth> padding_buffer = {0};
    memcpy(padding_buffer.data(), in, elements_left * sizeof(TIn));
    for (size_t i = elements_left; i < bitwidth; ++i)
      padding_buffer[i] = zero_point;
    pack_bitfield<bitpack_order>(padding_buffer.data(), out, zero_point);
  }
}



extern "C" void larq_ref_bsign(int8_t *input, uint32_t *output, size_t inputLength, int32_t zero_point)
{
  packbits_array<BitpackOrder::Canonical, std::int8_t, std::uint32_t>(input, inputLength, output, zero_point);
}

}  // namespace ref
}  // namespace compute_engine
