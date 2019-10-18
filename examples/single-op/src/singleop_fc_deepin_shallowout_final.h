#ifndef FC_DEEPIN_SHALLOWOUT_FINAL_H
#define FC_DEEPIN_SHALLOWOUT_FINAL_H

#define XC_FC_DEEPIN_SHALLOWOUT_FINAL_WEIGHTS 164, 196, 182, 172, 180, 202, 188, 180, 188, 185, 139, 181, 134, 169, 166, 195, 181, 196, 151, 175, 180, 189, 188, 157, 193, 184, 170, 191, 173, 186, 163, 198, 208, 240, 224, 216, 225, 245, 228, 220, 230, 228, 191, 227, 185, 220, 224, 246, 236, 246, 201, 225, 229, 241, 233, 200, 235, 228, 210, 229, 218, 227, 205, 238, 27, 44, 40, 42, 48, 64, 60, 54, 74, 62, 24, 69, 19, 44, 51, 77, 58, 63, 27, 68, 55, 82, 72, 50, 68, 66, 25, 43, 36, 44, 29, 43, 68, 81, 77, 87, 94, 118, 118, 106, 127, 112, 69, 115, 63, 85, 91, 119, 99, 104, 68, 109, 94, 125, 119, 103, 123, 121, 79, 93, 84, 84, 66, 74

#define XC_FC_DEEPIN_SHALLOWOUT_FINAL_BIASES -7348, 23643, 31596, 9456

#define XC_FC_DEEPIN_SHALLOWOUT_FINAL_SHIFT_SCALE 3, 3, 3, 3, 32051, 32051, 32051, 32051

typedef int8_t flatten_input_int8_t[1 * 32 * 1 * 1];
typedef int16_t xc_fc_deepin_shallowout_final_output_t[1 * 4];

void singleop_fc_deepin_shallowout_final(const flatten_input_int8_t *flatten_input_int8, xc_fc_deepin_shallowout_final_output_t *XC_fc_deepin_shallowout_final_output);

#endif /* FC_DEEPIN_SHALLOWOUT_FINAL_H */