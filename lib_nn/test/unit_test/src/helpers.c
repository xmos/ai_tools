#include <stdlib.h>
#include <stdint.h>

int pseudo_rand(int *seed){
  const int a = 1013904223;
  const int c = 1664525;
  *seed = (int)((long long)a * *seed + c);
  return *seed;
}

void pick_threshold_params(int32_t * thresholds, const unsigned chans_out, 
  const unsigned receptive_volume){
  for (int i = 0; i < chans_out; i++)
    thresholds[i] = i +  (int)receptive_volume/2 - (int)chans_out / 2;
}

//TODO pass the clamps
void pick_extreme_bias_post_activation_params(float * post_activation_multiplier, float * post_activation_bias, 
  unsigned chans_out, unsigned receptive_volume, int * seed){

  //The input range is from 0 to the receptive_volume (xor_popcount)
  float accu_min = 0; 
  float accu_max = receptive_volume*2; //the times 2 is due to the left shift in the output transform. 

  float input_range = accu_max - accu_min;

  float output_min = (float)INT8_MIN; 
  float output_max = (float)INT8_MAX; 
  float output_range = (output_max - output_min);

  for (unsigned ch = 0; ch < chans_out; ch++){
    float offset = 1024. * output_range * (float)pseudo_rand(seed)/(float)INT32_MAX;
    post_activation_multiplier[ch] = output_range / input_range;
    post_activation_bias[ch] = output_min - accu_min* output_range / input_range + offset;
  }
}
//TODO pass the clamps
void pick_extreme_mul_post_activation_params(float * post_activation_multiplier, float * post_activation_bias, 
  unsigned chans_out, unsigned receptive_volume, int * seed){

  //The input range is from 0 to the receptive_volume (xor_popcount)
  float accu_min = 0; 
  float accu_max = receptive_volume*2; //the times 2 is due to the left shift in the output transform. 

  float input_range = accu_max - accu_min;

  float output_min = (float)INT8_MIN; 
  float output_max = (float)INT8_MAX; 

  for (unsigned ch = 0; ch < chans_out; ch++){

    float output_overscale = 1024. + (float)pseudo_rand(seed)/(float)INT32_MAX;
    float output_range = (output_max - output_min)*output_overscale;
    post_activation_multiplier[ch] = output_range / input_range;
    post_activation_bias[ch] = output_min*output_overscale - accu_min* output_range / input_range;
  }
}

//TODO pass the clamps
void pick_post_activation_params(float * post_activation_multiplier, float * post_activation_bias, 
  unsigned chans_out, unsigned receptive_volume, int * seed){

/*
  std::int8_t Run(const std::int32_t accum, int out_channel) const {
    // Clamping is done in int32
    std::int32_t x = accum << 1;
    x = std::max<std::int32_t>(std::min<std::int32_t>(x, clamp_max), clamp_min);
    // The linear transformation is done in float
    float y =
        static_cast<float>(x) * multiplier[out_channel] + bias[out_channel];
    // And then we round back to int32 and clamp to the int8 range
    return saturate(round(y));
  }
*/

  //The input range is from 0 to the receptive_volume (xor_popcount)
  float accu_min = 0; 
  float accu_max = receptive_volume*2; //the times 2 is due to the left shift in the output transform. 

  float input_range = accu_max - accu_min;

  float output_min = (float)INT8_MIN; 
  float output_max = (float)INT8_MAX; 

  for (unsigned ch = 0; ch < chans_out; ch++){

    // Scale the input to extend beyond the range of the output such that we get some 
    // outputs that will saturate.
    float output_overscale = 0.5 + (float)pseudo_rand(seed)/(float)INT32_MAX;

    float output_range = (output_max - output_min)*output_overscale;

    // This offset allows the output range to completly miss the int8 output range sometimes
    float offset = 1.1 * output_range * (float)pseudo_rand(seed)/(float)INT32_MAX;

    post_activation_multiplier[ch] = output_range / input_range;
    post_activation_bias[ch] = output_min*output_overscale - accu_min* output_range / input_range + offset;
  }
}
