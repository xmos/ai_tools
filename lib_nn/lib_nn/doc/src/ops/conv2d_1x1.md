

# conv2d_1x1 #                                    {#oper_conv2d_1x1}


### Description 

This function performs a 2D convolution of an input image to produce an output image. 

@oper{conv2d_1x1} should be considered specialized version of @oper_ref{conv2d_deep}, for when the spatial dimensions of the convolution
window are @math{(1,1)}. More specifically, the operation performed by @oper{conv2d_1x1} is identical to that done by an instance of
@oper{conv2d_deep} when the following parameter constraints are applied to @oper{conv2d_deep}:

* @math{K_h = K_w = 1}   (a @math{1 \times 1} convolution window)
* @math{W_{vert} = W_{hori} = 1}  (both strides are @math{1}), and
* @math{W_{r0} = W_{c0} = 0}  (the convolution window starts at coordinates @math{(0,0)}

@note The parameters listed above are hyperparameters for @oper{conv2d_deep} and not for @oper{conv2d_1x1}, as their (implied) 
values are fixed in @oper{conv2d_1x1}.

Because of the implied constraints above, no input image padding is necessary (or possible) with this operator.

Conceptually, @oper{conv2d_1x1} can be thought of as performing a matrix-vector multiplication (with bias scales and offsets) to each 
input pixel to produce the corresponding output pixel. In this conception, the matrix is the kernel tensor @tensor{K}, the input 
vector's elements are the input channels for a given input pixel, and the output vector's elements are the output channels for the 
corresponding output pixel.

The @oper_ref{conv2d_deep}, @oper_ref{conv2d_shallowin}, and @oper_ref{conv2d_depthwise} operators are alternative 2D convolution operators
optimized for different circumstances.

### Parameters 

#### Hyperparameters        {#conv2d_1x1_hyperparams}

The following are the hyperparameters of @oper{conv2d_1x1}.

@par

<table>
<tr><th>Symbol(s)                   <th>Description

<tr><td>@tensor_shape{X_h, X_w}     <td>The height and width of both the input and output images.
<tr><td>@math{X_c}                  <td>The input image channel count.
<tr><td>@math{Y_c}                  <td>The output image channel count.
</table>

@note Because of the implied convolution window parameters, the spatial dimensions of the input and output images are the same.

##### Hyperparameter Constraints        {#conv2d_1x1_hyperparm_constraints}

* The input and output channel counts must be a multiple of `4`.
  * @math{ X_c = 0 \left( \text{mod } 4 \right) }
  * @math{ Y_c = 0 \left( \text{mod } 4 \right) }

#### Data Parameters

The following are input and output parameters of @oper{conv2d_1x1}.

@par

<table>
<tr><th colspan="2">Symbol          <th>Direction   <th>Shape                       <th>Description
<tr><td colspan="2">@tensor{Y}      <td>out         <td>@math{(X_h, X_w, Y_c)}      <td>The output image.
<tr><td colspan="2">@tensor{X}      <td>in          <td>@math{(X_h, X_w, X_c)}      <td>The input image.
<tr><td colspan="2">@tensor{K}      <td>in          <td>@math{(Y_c, K_h, K_w, X_c)} <td>The kernel tensor.
<tr><td colspan="2">[`BSO`]         <td>in          <td>                            <td>The elements of the bias-scale-offset array (see @ref out_shift_scale).
<tr><td>        <td>@tensor{B}      <td>            <td>@math{Y_c}                  <td>The output channel biases.        
<tr><td>        <td>@tensor{s_1}    <td>            <td>@math{Y_c}                  <td>The first output channel shifts.
<tr><td>        <td>@tensor{s_2}    <td>            <td>@math{Y_c}                  <td>The output channel scales.
<tr><td>        <td>@tensor{o_a}    <td>            <td>@math{Y_c}                  <td>The output channel offset scales.
<tr><td>        <td>@tensor{o_b}    <td>            <td>@math{Y_c}                  <td>The output channel offset values.
<tr><td>        <td>@tensor{s_3}    <td>            <td>@math{Y_c}                  <td>The final output channel shifts.
</table>


### Operation Performed

@f[
     V\left[r,c,p\right]=
         B_p+\sum_{k=0}^{X_c-1} 
         X\left[ r,c,k\right]\cdot K\left[p,k\right]\\\
  \\\  
      Y\left[r,c,p\right]= sat_{8}\left(\frac{\left(sat_{16}\left(\frac{V\left[r,c,p\right]}
             {2^{s_{1p}}}\right)\cdot s_{2p}\right)}{2^{s_{3p}}}\right)
@f]

where  

@par
@tensor{V} is an intermediate tensor (holding the 32-bit accumulators),

@par
@math{(r,c,p)} are the output row, column and channel,

@par
@math{sat_8\left(\cdot\right)} and @math{sat_{16}\left(\cdot\right)} saturate their arguments 
     to @math{8}- and @math{16}-bit bounds, and
@par
the remaining parameters are as described above.

### Example Diagram

@todo Create diagram

### Configuration Options

The following sections describe configurable options for the @oper{conv2d_1x1} operator. Configuration options can
be set by adding the appropriate project-wide build flag.

#### CONFIG_SYMMETRIC_SATURATION_conv2d_1x1

By default, when the @oper{conv2d_1x1} operator applies saturation logic to its output, it uses the standard bounds 8-bit 
signed integers, namely `(-128, 127)`. However, the XS3 VPU hardware is designed to apply symmetric saturation bounds of
`(-127,127)`. Using the asymmetric bounds for this operation is more expensive, but necessary for some applications.

If either the preprocessor symbol `CONFIG_SYMMETRIC_SATURATION_conv2d_1x1` is defined to `1`, or if 
`CONFIG_SYMMETRIC_SATURATION_conv2d_1x1` is undefined, but `CONFIG_SYMMETRIC_SATURATION_GLOBAL` is defined to `1`,
then the symmetric saturation bounds will be used instead. 

Note that this option *only* affects the output saturation bounds. It does *not* affect the saturation bounds of the
intermediate 32-bit or 16-bit accumulators.

See nn_config.h.

