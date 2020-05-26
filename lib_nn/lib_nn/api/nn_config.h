#pragma once

/**
 * @macro CONFIG_SYMMETRIC_SATURATION_GLOBAL
 * @brief Configure whether (supported) operators use `-127` or `-128` as the their lower saturation bound.
 * 
 * The output of 8-bit arithmetic on the XS3 VPU has natural symmetric saturation bounds of (`-127`, `127`). This may be
 * unacceptable, in which case (`-128`, `127`) can be used instead.
 * 
 * If `CONFIG_SYMMETRIC_SATURATION_GLOBAL` is defined, it is used as the value for each config macro 
 * `CONFIG_SYMMETRIC_SATURATION_*` (e.g. `CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8`), unless 
 * that macro has been explicitly set.
 * 
 * Bypassing the symmetric saturation bound requires additional logic, and so will generally make the
 * operators slower, though this will be more or less significant, depending on the specific
 * operators.
 */


/**
 * @macro CONFIG_SYMMETRIC_SATURATION_conv2d_deep
 * @brief Configure whether `-127` or `-128` is used as the saturation limit for `conv2d_deep()`.
 * 
 * The output of 8-bit arithmetic on the XS3 VPU has natural symmetric saturation bounds of (`-127`, `127`). This may be
 * unacceptable, in which case (`-128`, `127`) can be used instead.
 * 
 * To specify that the symmetric saturation lower bound (`-127`) should be used for `conv2d_deep()`, define 
 * `CONFIG_SYMMETRIC_SATURATION_conv2d_deep` to be `1`. If it is defined to `0`, `-128` will be used instead.
 * 
 * If `CONFIG_SYMMETRIC_SATURATION_conv2d_deep` is undefined, then the value of `CONFIG_SYMMETRIC_SATURATION_GLOBAL`
 * is used instead, if that is defined. If neither symbol is defined, `CONFIG_SYMMETRIC_SATURATION_conv2d_deep()`
 * defaults to 0, using a lower saturation bound of `-128`.
 * 
 */
#ifndef CONFIG_SYMMETRIC_SATURATION_conv2d_deep
  #ifdef CONFIG_SYMMETRIC_SATURATION_GLOBAL
    #define CONFIG_SYMMETRIC_SATURATION_conv2d_deep CONFIG_SYMMETRIC_SATURATION_GLOBAL
  #else
    #define CONFIG_SYMMETRIC_SATURATION_conv2d_deep (0)
  #endif
#endif 


/**
 * @macro CONFIG_SYMMETRIC_SATURATION_conv2d_shallowin
 * @brief Configure whether `-127` or `-128` is used as the saturation limit for `conv2d_shallowin()`.
 * 
 * The output of 8-bit arithmetic on the XS3 VPU has natural symmetric saturation bounds of (`-127`, `127`). This may be
 * unacceptable, in which case (`-128`, `127`) can be used instead.
 * 
 * To specify that the symmetric saturation lower bound (`-127`) should be used for `conv2d_shallowin()`, define 
 * `CONFIG_SYMMETRIC_SATURATION_conv2d_shallowin` to be `1`. If it is defined to `0`, `-128` will be used instead.
 * 
 * If `CONFIG_SYMMETRIC_SATURATION_conv2d_shallowin` is undefined, then the value of `CONFIG_SYMMETRIC_SATURATION_GLOBAL`
 * is used instead, if that is defined. If neither symbol is defined, `CONFIG_SYMMETRIC_SATURATION_conv2d_shallowin()`
 * defaults to 0, using a lower saturation bound of `-128`.
 * 
 */
#ifndef CONFIG_SYMMETRIC_SATURATION_conv2d_shallowin
  #ifdef CONFIG_SYMMETRIC_SATURATION_GLOBAL
    #define CONFIG_SYMMETRIC_SATURATION_conv2d_shallowin CONFIG_SYMMETRIC_SATURATION_GLOBAL
  #else
    #define CONFIG_SYMMETRIC_SATURATION_conv2d_shallowin (0)
  #endif
#endif 

#ifndef CONFIG_SYMMETRIC_SATURATION_conv2d_im2col
  #ifdef CONFIG_SYMMETRIC_SATURATION_GLOBAL
    #define CONFIG_SYMMETRIC_SATURATION_conv2d_im2col CONFIG_SYMMETRIC_SATURATION_GLOBAL
  #else
    #define CONFIG_SYMMETRIC_SATURATION_conv2d_im2col (0)
  #endif
#endif 

/**
 * @macro CONFIG_SYMMETRIC_SATURATION_conv2d_depthwise
 * @brief Configure whether `-127` or `-128` is used as the saturation limit for `conv2d_depthwise()`.
 * 
 * The output of 8-bit arithmetic on the XS3 VPU has natural symmetric saturation bounds of (`-127`, `127`). This may be
 * unacceptable, in which case (`-128`, `127`) can be used instead.
 * 
 * To specify that the symmetric saturation lower bound (`-127`) should be used for `conv2d_depthwise()`, define 
 * `CONFIG_SYMMETRIC_SATURATION_conv2d_depthwise` to be `1`. If it is defined to `0`, `-128` will be used instead.
 * 
 * If `CONFIG_SYMMETRIC_SATURATION_conv2d_depthwise` is undefined, then the value of `CONFIG_SYMMETRIC_SATURATION_GLOBAL`
 * is used instead, if that is defined. If neither symbol is defined, `CONFIG_SYMMETRIC_SATURATION_conv2d_depthwise()`
 * defaults to 0, using a lower saturation bound of `-128`.
 * 
 */
#ifndef CONFIG_SYMMETRIC_SATURATION_conv2d_depthwise
  #ifdef CONFIG_SYMMETRIC_SATURATION_GLOBAL
    #define CONFIG_SYMMETRIC_SATURATION_conv2d_depthwise CONFIG_SYMMETRIC_SATURATION_GLOBAL
  #else
    #define CONFIG_SYMMETRIC_SATURATION_conv2d_depthwise (0)
  #endif
#endif 


/**
 * @macro CONFIG_SYMMETRIC_SATURATION_conv2d_1x1
 * @brief Configure whether `-127` or `-128` is used as the saturation limit for `conv2d_1x1()`.
 * 
 * The output of 8-bit arithmetic on the XS3 VPU has natural symmetric saturation bounds of (`-127`, `127`). This may be
 * unacceptable, in which case (`-128`, `127`) can be used instead.
 * 
 * To specify that the symmetric saturation lower bound (`-127`) should be used for `conv2d_1x1()`, define 
 * `CONFIG_SYMMETRIC_SATURATION_conv2d_1x1` to be `1`. If it is defined to `0`, `-128` will be used instead.
 * 
 * If `CONFIG_SYMMETRIC_SATURATION_conv2d_1x1` is undefined, then the value of `CONFIG_SYMMETRIC_SATURATION_GLOBAL`
 * is used instead, if that is defined. If neither symbol is defined, `CONFIG_SYMMETRIC_SATURATION_conv2d_1x1()`
 * defaults to 0, using a lower saturation bound of `-128`.
 * 
 */
#ifndef CONFIG_SYMMETRIC_SATURATION_conv2d_1x1
  #ifdef CONFIG_SYMMETRIC_SATURATION_GLOBAL
    #define CONFIG_SYMMETRIC_SATURATION_conv2d_1x1 CONFIG_SYMMETRIC_SATURATION_GLOBAL
  #else
    #define CONFIG_SYMMETRIC_SATURATION_conv2d_1x1 (0)
  #endif
#endif 


/**
 * @macro CONFIG_SYMMETRIC_SATURATION_avgpool2d
 * @brief Configure whether `-127` or `-128` is used as the saturation limit for `avgpool2d()`.
 * 
 * The output of 8-bit arithmetic on the XS3 VPU has natural symmetric saturation bounds of (`-127`, `127`). This may be
 * unacceptable, in which case (`-128`, `127`) can be used instead.
 * 
 * To specify that the symmetric saturation lower bound (`-127`) should be used for `avgpool2d()`, define 
 * `CONFIG_SYMMETRIC_SATURATION_avgpool2d` to be `1`. If it is defined to `0`, `-128` will be used instead.
 * 
 * If `CONFIG_SYMMETRIC_SATURATION_avgpool2d` is undefined, then the value of `CONFIG_SYMMETRIC_SATURATION_GLOBAL`
 * is used instead, if that is defined. If neither symbol is defined, `CONFIG_SYMMETRIC_SATURATION_avgpool2d()`
 * defaults to 0, using a lower saturation bound of `-128`.
 * 
 */
#ifndef CONFIG_SYMMETRIC_SATURATION_avgpool2d
  #ifdef CONFIG_SYMMETRIC_SATURATION_GLOBAL
    #define CONFIG_SYMMETRIC_SATURATION_avgpool2d CONFIG_SYMMETRIC_SATURATION_GLOBAL
  #else
    #define CONFIG_SYMMETRIC_SATURATION_avgpool2d (0)
  #endif
#endif 



/**
 * @macro CONFIG_SYMMETRIC_SATURATION_avgpool2d_global
 * @brief Configure whether `-127` or `-128` is used as the saturation limit for `avgpool2d_global()`.
 * 
 * The output of 8-bit arithmetic on the XS3 VPU has natural symmetric saturation bounds of (`-127`, `127`). This may be
 * unacceptable, in which case (`-128`, `127`) can be used instead.
 * 
 * To specify that the symmetric saturation lower bound (`-127`) should be used for `avgpool2d_global()`, define 
 * `CONFIG_SYMMETRIC_SATURATION_avgpool2d_global` to be `1`. If it is defined to `0`, `-128` will be used instead.
 * 
 * If `CONFIG_SYMMETRIC_SATURATION_avgpool2d_global` is undefined, then the value of `CONFIG_SYMMETRIC_SATURATION_GLOBAL`
 * is used instead, if that is defined. If neither symbol is defined, `CONFIG_SYMMETRIC_SATURATION_avgpool2d_global()`
 * defaults to 0, using a lower saturation bound of `-128`.
 * 
 */
#ifndef CONFIG_SYMMETRIC_SATURATION_avgpool2d_global
  #ifdef CONFIG_SYMMETRIC_SATURATION_GLOBAL
    #define CONFIG_SYMMETRIC_SATURATION_avgpool2d_global CONFIG_SYMMETRIC_SATURATION_GLOBAL
  #else
    #define CONFIG_SYMMETRIC_SATURATION_avgpool2d_global (0)
  #endif
#endif 


/**
 * @macro CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8
 * @brief Configure whether `-127` or `-128` is used as the saturation limit for `requantize_16_to_8()`.
 * 
 * The output of 8-bit arithmetic on the XS3 VPU has natural symmetric saturation bounds of (`-127`, `127`). This may be
 * unacceptable, in which case (`-128`, `127`) can be used instead.
 * 
 * To specify that the symmetric saturation lower bound (`-127`) should be used for `requantize_16_to_8()`, define 
 * `CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8` to be `1`. If it is defined to `0`, `-128` will be used instead.
 * 
 * If `CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8` is undefined, then the value of `CONFIG_SYMMETRIC_SATURATION_GLOBAL`
 * is used instead, if that is defined. If neither symbol is defined, `CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8()`
 * defaults to 0, using a lower saturation bound of `-128`.
 * 
 * Unfortunately, bypassing the symmetric saturation bounds requires significant additional logic, and so with the symmetric 
 * saturation bound, `requantize_16_to_8()` is approximately 2.5x faster.
 * 
 */
#ifndef CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8
  #ifdef CONFIG_SYMMETRIC_SATURATION_GLOBAL
    #define CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8 CONFIG_SYMMETRIC_SATURATION_GLOBAL
  #else
    #define CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8 (0)
  #endif
#endif 