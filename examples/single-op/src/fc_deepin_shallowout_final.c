#include "nn_operator.h"
#include "fc_deepin_shallowout_final.h"

#ifdef __XC__
#define WORD_ALIGNED [[aligned(4)]]
#else
#define WORD_ALIGNED
#endif

const int8_t WORD_ALIGNED XC_fc_deepin_shallowout_final_weights[4 * 32] = {XC_FC_DEEPIN_SHALLOWOUT_FINAL_WEIGHTS};
const int32_t WORD_ALIGNED XC_fc_deepin_shallowout_final_biases[4] = {XC_FC_DEEPIN_SHALLOWOUT_FINAL_BIASES};
const int16_t WORD_ALIGNED XC_fc_deepin_shallowout_final_shift_scale[2 * 4] = {XC_FC_DEEPIN_SHALLOWOUT_FINAL_SHIFT_SCALE};

void fc_deepin_shallowout_final(const flatten_input_int8_t *flatten_input_int8, xc_fc_deepin_shallowout_final_output_t *XC_fc_deepin_shallowout_final_output)
{

     fc_deepin_shallowout_lin(XC_fc_deepin_shallowout_final_weights, XC_fc_deepin_shallowout_final_biases, (int8_t *)flatten_input_int8, (int16_t *)XC_fc_deepin_shallowout_final_output, 4, 32, (uint16_t*) &XC_fc_deepin_shallowout_final_shift_scale[0], (int16_t*) &XC_fc_deepin_shallowout_final_shift_scale[4]);
}
