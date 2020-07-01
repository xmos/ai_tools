
Notes for lib_nn                          {#notes}
================

### Note 1: Saturation, Accumulation, Shifts and Scales ###                {#sat_acc_shift_scale}


For the convolution and fully-connected layers, it is important to understand the 
behavior of the XS3 VPU.

Each output comprises a series of 8-bit multiplies which accumulate into a 32-bit 
accumulator -- effectively the 32-bit dot-product of two 8-bit vectors. Prior to
this dot product, the accumulator is seeded with a 32-bit bias value.

While accumulating, if at any point the sum would go beyond the range `(2^-31 + 1, 2^31-1)`,
the accumulator will saturate to `-2^31 + 1` if the sum is negative and `2^31-1` if 
it's positive. The accumulator does not roll-over.

After 32-bit accumulation is completed the following occurs:
- The output has a user-specified arithmetic right bit-shift applied (rounded).
- The result is saturated to 16-bit bounds (2^15+1, 2^15-1) if necessary.
- The result is quantized to 16 bits by discarding the upper half word.

After shifting a scale factor is applied to each output, which consists of:
- Multiplying the 16-bit result by a 16-bit signed integer
- Applying an arithmetic right shift of 14 bits to the 30-bit result (rounded).
- Saturating the result to 16-bit bounds.
(This can be thought of as a rounding, saturating fixed-point multiplication in which
the 16-bit accumulator value is treated as a signed Q15.0 fixed-point value and the
scale factor is treated as a signed Q1.14 fixed-point value).

For `fc_deepin_shallowout_16()` the results are then stored in the output.

For the 2D convolution functions, the final step before storing the result is to
quantize the 16-bit results down to signed 8-bit results by right-shifting 8 bits 
and rounding.


### Note 2: Standard Tensor Layout ###      {#standard_layout}


The standard layout of an N-dimensional tensor with shape `(s1, s2, ..., sN)` is the
same as the typical layout of a C array with the same dimensions. In this layout, the
as memory address increases, indices of each dimension are iterated over, with indices 
iterating in ascending order, and with the later dimensions iterating fastest.

For example, a tensor `A` with shape (2,2,2), in standard tensor layout would be ordered
as:

`A[0,0,0], A[0,0,1], A[0,1,0], A[0,1,1], A[1,0,0], A[1,0,1], A[1,1,0], A[1,1,1]`



### Note 3: Inner Products and Saturation ###      {#inner_prod_sat}

 Many functions in this API compute inner products between vectors with many elements. These
 inner products are computed as long sequences of multiply-accumulates on the VPU. Unlike on
 the scalar unit, on the VPU multiplications, additions and subtractions *are not associative*. 

 The lack of associativity is due to the saturation logic used on the VPU. Where the scalar
 unit will roll-over in the case of integer overflows, the XS3 VPU will clamp results
 to the bounds appropriate to the element bit-depth, which, for N-bit (signed) integers,
 is the symmetric range  [-(2^(N-1))+1, (2^(N-1))-1]. Macros are provided for convenience.

    Saturation Bounds:

        Bit depth   Min             Min Macro           Max             Max Macro
        =========   ===             =========           ===             =========
        8-bit:      -127            VPU_INT8_MIN        127             VPU_INT8_MAX
        16-bit:     -65535          VPU_INT16_MIN       65535           VPU_INT16_MAX
        32-bit:     -2147483647     VPU_INT32_MIN       2147483647      VPU_INT32_MAX


When computing inner products, saturation occurs based on the *accumulator* bit depth, rather than
the multiplicand (vector element) bit depth.

        Element         Accumulator
        =======         ===========
        8-bit           32-bit
        16-bit          32-bit
        32-bit          40-bit

Most inner products computed in this API use 8-bit input vectors. The product of two 8-bit signed
integers can be no larger in magnitude than 2^14  (from -(2^7)*-(2^7) ). The largest 32-bit accumulator
value is approximately 2^31, which is (2^14 * 2^17). Thus, an 8-bit inner product cannot 
saturate its 32-bit accumulator unless the vectors are about 128,000 elements long.

However, when inner products are computed, the accumulator is (usually) seeded with an arbitrary
user-supplied 32-bit bias. This bias makes it possible for saturation to occur on inner products
with operand vectors of any length.

Further, saturation can occur at any point during the accumulation, and subsequent steps may
move the accumulator away from the saturation point, and so it may not always be obvious whether
saturation has occurred somewhere inside the inner product, skewing the final result.

Finally, *the functions in this API neither specify nor guarantee the order in which elements
are accumulated when computing inner products*.

Therefore, where saturation in unacceptable, it is incumbent upon the *user* of this library to 
ensure that saturation is not possible given the inputs (matrix/kernel coefficients and input
vectors) and other parameters (e.g. input channel count).

### Note 4: Output Shifts and Scale ###      {#out_shift_scale}

Many functions in this API include shifts and a scale on each output prior to writing the result
to memory. For the sake of brevity, the details of these operations are contained here, rather than 
repeating them in each function's description.

In general, the situation looks like this:

        y[i] <- ((acc32[i] >> shr1[i]) * scale[i]) >> shr2[i]             (16-bit outputs)

            or
        
        y[i] <- (((acc32[i] >> shr1[i]) * scale[i]) >> shr2[i]) >> 8        (8-bit outputs)

      where
        i is the index of the output
        y[i] is the either 8- or 16-bit output
        acc32[i] is the 32-bit accumulator value associated with output i
            (acc32[i] is an intermediate value, which may be the result of convolution or an inner product, etc)
        shr1[i] is the first shift value associated with output i
        scale[i] is the scale value associated with output i
        shr2[i] is teh second shift value associated with output i

Shift 1:
    The shift operation performs several actions atomically:
        - First a "virtual" arithmetic right shift of `shr` bits occurs on the 32-bit accumulator `acc32`.
        - Second, the result of the shift is rounded (as though the shift is a rounding real division by `2^-shr`)
        with ties rounding towards positive infinity.
        - Third, saturation logic is applied to the result of the rounding, clamping the result to [-65535, 65535]. Note
        that this saturation is symmetric.
        - Finally, the bit-depth of the result is reduced to 16 bits.
    While `shr` is a signed 16-bit value, *negative values will be treated as zero*.
    As a final ideosyncrasy, the shifting of negative accumulator values will *never result in zeros*. Where
    the result of shifting a negative value would normally underflow to 0, it will instead result as -1.

Scale:
    The scaling is a signed 16-bit integer multiplication for a 32-bit (actual max value is ((2^15)-1)^2). 

Shift 2:
    This behaves exactly like shift 1, but is applied after the scale.

Final shift:
    In functions that output 8-bit results, a final shift of 8 bits is applied after shift 2 to
    reduce the bit-depth from 16 bits to 8 bits. In this case, rounding occurs, as with the other
    two operations, but no saturation is possible.
    In functions that output 16-bit results, no final shift occurs here.


### Note 5: Bias-Scale-Offset Tensor Layout ###      {#bso_layout}

In most cases, where a function requires a Bias-Scale-Offset (BSO) tensor as input, the required layout will
be as specified here [0].

The BSO tensor contains information pertaining to the `C_out` output channels of an operation. A BSO tensor
is 3 dimensional with shape (`ceil(C_out/16.0)`, `7`, `16`) with elements of type `data16_t`.

The first axis corresponds to output channel groups. The `ceil( )` is applied because for the purposes of
a BSO tensor, the output channel tail must be handled as a full output channel group. (see @ref c_groups)

The third axis corresponds to the output channel offset within the output channel group. All information
corresponding to output channel `k` will be found in the elements `BSO[(k//16), :, (k%16)]`.

The second axis corresponds to the specific parameter. The indices are:

    0 - Bias high half-word: The most significant 16-bits of the 32-bit bias for the output channel
    1 - Bias low half-word: The least significant 16-bits of the 32-bit bias for the output channel
    2 - Shift1: The first right-shift value; applied to the 32-bit accumulator for a 16-bit result.
    3 - Scale: The scale value; applied after shift1; multiplied by the 16-bit result for a 32-bit product
    4 - Offset Scale: Scale for Offset
    5 - Offset: Multiplied by Offset Scale; product is accumulated into the 32-bit accumulator
    6 - Shift2: The second right-shift value; applied after scale; down-shifts the 32-bit accumulator for an 8-bit result.

[0]     For general information about how shifts and scales are used, see "Notes on Output Shifts and Scale", 
        for information on how biases are used, see "Notes on Inner Products and Saturation"
[1]     See "Notes on Channel Output and Input Groups".

### Note 6: Channel Output and Input Groups ###      {#c_groups}

Most functions in this API are capable of handling many input and output channels. The XS3 VPU is capable
of consuming many input channels, or producing many output channels in a single operation. These lead to the
concept of input channel groups and output channel groups being baked deeply into this API.

Most functions in this API operate on 8-bit inputs and produce 8-bit outputs [0]. The XS3 vector register size is 256
bits, so in 8-bit mode, the XS3 VPU is capable of loading (and operating on) 32 8-bit values simultaneously [1]. When
the operation is a vector multiply-accumulate, it is desirable to have accumulators which are significantly larger 
than the largest possible product of two 8-bit signed integers (`2^14`). To this end, XS3 uses 32-bit accumulators 
for 8-bit mode [2], spread across the `vD` and `vR` vector registers [3]. That leaves room for 16 accuulators in 8-bit 
mode.

The `xs3_vpu.h` header file contains macros reflecting the number of elements and number of accumulators in each
operating mode. In particular, `VPU_INT8_EPV` is 32, and `VPU_INT8_ACC_PERIOD` is 16 [4].

When the documentation for a function in this API has `C_in` input channels and `C_out` output channels, the function
may require that the parameter tensors consumed by that function be formatted in a layout which reflects the 
channel groups. In most cases, the number of input channel groups will be `ceil(C_in / 32.0)` -- the number of `VLMACCR`
instructions required to multiply-accumulate all input channels -- and the number of output channel groups will be 
`ceil(C_out / 16.0)` -- the number of outputs that can be simultaneously calculated. 

Where tensors are layed out in this way, output channel group 'k' refers to output channels `16*k` to `16*k+15` (inclusive).
Likewise, input channel group `k` refers to input channels `32*k` to `32*k+31` (inclusive). Some tensors will be arranged 
such that one of the dimensions correponds, for example, to the ouput channel group, and another dimension will refer to
the output channel offset. That output (or input) offset is just the offset index *within* the channel group of that channel
(i.e. for output channel `c`, `c % 16`).

Where the number of output (input) channels is not a multiple of 16 (32) --and where the function allows this -- there will be 
an ouput (input) channel tail. The tail are the last channels which do not form complete group. Some tensors, in particular 
the bias-shifts-scale tensors, will require that tails be padded. Whether padding must be zeros, or if it is safe to use 
arbitrary values is specified by the function.

[0]     Some functions work on elements that are not 8 bits, and most functions operate on intermediate values of
        16 or 32 bits.
[1]     Most instructionss in 8-bit mode that load elements from memory(VLMACCR, VLMUL, VLDR, VLADD, etc) load 32 8-bit
        values. The main exception to this is the VLMACC instruction, which loads only 16 8-bit values.
[2]     16-bit mode also uses 32-bit accumulators (and so `VPU_INT8_ACC_PERIOD == VPU_INT16_ACC_PERIOD`), 32-bit mode
        uses 40-bit accumulators.
[3]     In 8- and 16-bit modes, `vD` holds the 16 most significant bits of the accumulator and `vR` holds the 16 least
        significant bits.
[4]     "EPV" is elements per vector. "ACC_PERIOD" is the accumulator period. The accumulators have a "period" because of
        the `VLMACCR` instruction, which adds a 32-element (in 8-bit mode) inner product between vector regiser `vC` and a
        sequence of values in memory to a single accumulator, and then does a cyclic rotation of the accumulators. Then, in 
        8-bit mode 16 (`VPU_INT8_ACC_PERIOD`) `VLMACCR` instructions will perform a full rotation of the accumulators.
