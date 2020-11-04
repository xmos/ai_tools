

# conv2d_depthwise #                                    {#oper_conv2d_depthwise}


### Description 

This operator performs a 2D convolution of an input image to produce an output image. The convolution is considered 
"depthwise" when there is no interaction between input channel @math{j} and output channel @math{k} unless @math{j=k}.

@oper{conv2d_depthwise} should be considered specialized version of @oper_ref{conv2d_deep}, optimized for when the 2D 
sub-tensors given by @math{K[:,r,c,:]} are diagonal (and square) matrices. In such a circumstance, the off-diagonal 
elements can be omitted and the computation can be performed more quickly.

This operator supports implied padding of the input image in which a specified value (@math{z_0}) is used for all 
padding channels.

The @oper_ref{conv2d_deep}, @oper_ref{conv2d_1x1}, and @oper_ref{conv2d_shallowin} operators are alternative 2D 
convolution operators specialized for different circumstances.

### Parameters 

#### Hyperparameters        {#conv2d_depthwise_hyperparams}

The following are the hyperparameters of @oper{conv2d_depthwise}.

@par

<table>
<tr><th>Symbol(s)       <th>Description

<tr><td>@tensor_shape{X_h, X_w}         <td>The input image height and width, respectively.
<tr><td>@tensor_shape{Y_h, Y_w}         <td>The output image height and width, respectively.
<tr><td>@math{X_c}                      <td>The channel count of both the input and output images.
<tr><td>@tensor_shape{K_h, K_w}         <td>The convolution window height and width, respectively.
<tr><td>@math{(W_{r0}, W_{c0})}         <td>The convolution window's start position, which are the coordinates of the 
                                            upper-left pixel of the convolution window (in the input image's coordinate 
                                            space) corresponding to the output pixel at @math{(0,0)} (in the output 
                                            image's coordinate space).
<tr><td>@math{(W_{vert}, W_{hori})}     <td>The convolution window's vertical and horizontal strides, respectively.
<tr><td>@math{z_0}                      <td>The zero-point value, used for padding pixels.
</table>

@note In the future a "depth multiplier" may be supported. A depth multiplier is the (integer) ratio of output channel 
    count to input channel count. With a depth multiplier of `N`, each input channel contributes to `N` different output 
    channels (though it is still true that no two input channels contribute to the same output channel). Currently, the 
    depth multiplier has an implicit value of `1`.


##### Hyperparameter Constraints        {#conv2d_depthwise_hyperparm_constraints}

* The input (and output) channel count must be a multiple of `4`.
  * @math{ X_c = 0 \left( \text{mod } 4 \right) }
* The convolution window must never be entirely in padding.
 * @math{ W_{r0} + K_h > 0 }
 * @math{ W_{c0} + K_w > 0 }
 * @math{ W_{r0} + W_{vert} * \left( Y_h - 1 \right)  \lt X_h }
 * @math{ W_{c0} + W_{hori} * \left( Y_w - 1 \right)  \lt X_w }


#### Data Parameters

The following are input and output data parameters of @oper{conv2d_depthwise}.

@par

<table>
<tr><th colspan="2">Symbol          <th>Direction   <th>Shape                       <th>Description
<tr><td colspan="2">@tensor{Y}      <td>out         <td>@math{(Y_h, Y_w, X_c)}      <td>The output image.
<tr><td colspan="2">@tensor{X}      <td>in          <td>@math{(X_h, X_w, X_c)}      <td>The input image.
<tr><td colspan="2">@tensor{K}      <td>in          <td>@math{(K_h, K_w, X_c)}      <td>The kernel tensor.
<tr><td colspan="2">[`BSO`]         <td>in          <td>                            <td>The elements of the 
                                                                                        bias-scale-offset array (see 
                                                                                        @ref out_shift_scale).
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
         B_p+\sum_{w_r=0}^{K_h-1}\sum_{w_c=0}^{K_w-1} 
        \hat X\left[ W_{r0}+r\cdot W_{vert}+w_r,
                 W_{c0}+c\cdot W_{hori}+w_c,
                 p\right]\cdot K\left[w_r,w_c,p\right]\\\
  \\\  
      Y\left[r,c,p\right]= sat_{8}\left(\frac{\left(sat_{16}\left(\frac{V\left[r,c,p\right]}
             {2^{s_{1p}}}\right)\cdot s_{2p}\right)}{2^{s_{3p}}}\right)
@f]

where  

@par
@math{\hat X[i,j,k]} takes the value @math{X[i,j,k]} when the indices are within the input image's bounds,
and takes the value @math{z_0} otherwise,

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

The following sections describe configurable options for the @oper{conv2d_depthwise} operator. Configuration options can
be set by adding the appropriate project-wide build flag.

#### CONFIG_SYMMETRIC_SATURATION_conv2d_depthwise

By default, when the @oper{conv2d_depthwise} operator applies saturation logic to its output, it uses the standard bounds 8-bit 
signed integers, namely `(-128, 127)`. However, the XS3 VPU hardware is designed to apply symmetric saturation bounds of
`(-127,127)`. Using the asymmetric bounds for this operation is more expensive, but necessary for some applications.

If either the preprocessor symbol `CONFIG_SYMMETRIC_SATURATION_conv2d_depthwise` is defined to `1`, or if 
`CONFIG_SYMMETRIC_SATURATION_conv2d_depthwise` is undefined, but `CONFIG_SYMMETRIC_SATURATION_GLOBAL` is defined to `1`,
then the symmetric saturation bounds will be used instead. 

Note that this option *only* affects the output saturation bounds. It does *not* affect the saturation bounds of the
intermediate 32-bit or 16-bit accumulators.

See nn_config.h.

