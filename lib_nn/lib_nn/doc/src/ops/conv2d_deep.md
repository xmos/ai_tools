

# conv2d_deep #                                     {#oper_conv2d_deep}


### Description 

This operator performs a 2D convolution of an input image to produce an output image. The convolution is considered "deep" 
when the input and output images each have many channels.

The @oper_ref{conv2d_shallowin}, @oper_ref{conv2d_1x1}, and @oper_ref{conv2d_depthwise} operators are alternative 2D convolution operators
optimized for special circumstances.

This operator supports implied padding of the input image in which a specified value (@math{z_0}) is used for all padding channels.

### Parameters 

#### Hyperparameters        {#conv2d_deep_hyperparams}

The following are the hyperparameters of @oper{conv2d_deep}.

@par

<table>
<tr><th>Symbol(s)       <th>Description

<tr><td>@tensor_shape{X_h, X_w, X_c}    <td>The input image dimensions, which correspond to height, width and channel count respectively.
<tr><td>@tensor_shape{Y_h, Y_w, Y_c}    <td>The output image dimensions, which correspond to height, width and channel count respectively.
<tr><td>@tensor_shape{K_h, K_w}         <td>The convolution window height and width, respectively.
<tr><td>@math{(W_{r0}, W_{c0})}         <td>The convolution window's start position, which are the coordinates of the upper-left pixel of the 
                                            convolution window (in the input image's coordinate space) corresponding to the output pixel at
                                            @math{(0,0)} (in the output image's coordinate space).
<tr><td>@math{(W_{vert}, W_{hori})}     <td>The convolution window's vertical and horizontal strides, respectively.
<tr><td>@math{z_0}                      <td>The zero-point value, used for padding pixels.
</table>

##### Hyperparameter Constraints

* The input and output channel counts must be a multiple of `4`.
  * @math{ X_c = 0 \left( \text{mod } 4 \right) }
  * @math{ Y_c = 0 \left( \text{mod } 4 \right) }
* The convolution window must never be entirely in padding.
 * @math{ W_{r0} + K_h > 0 }
 * @math{ W_{c0} + K_w > 0 }
 * @math{ W_{r0} + W_{vert} * \left( Y_h - 1 \right)  \lt X_h }
 * @math{ W_{c0} + W_{hori} * \left( Y_w - 1 \right)  \lt X_w }

#### Data Parameters

The following are input and output parameters of @oper{conv2d_deep}.

@par

<table>
<tr><th colspan="2">Symbol          <th>Direction   <th>Shape                       <th>Description
<tr><td colspan="2">@tensor{Y}      <td>out         <td>@math{(Y_h, Y_w, Y_c)}      <td>The output image.
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
         B_p+
         \sum_{w_r=0}^{K_h-1}\sum_{w_c=0}^{K_w-1}\sum_{k=0}^{X_c-1} 
         \hat X\left[ W_{r0}+r\cdot W_{vert}+w_r,
                 W_{c0}+c\cdot W_{hori}+w_c,
                 k\right]\cdot K\left[p,w_r,w_c,k\right]\\\
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


 The following diagram shows an example of a @math{3 \times 3} convolution window moving across an input image with shape 
 @math{5 \times 7}, with vertical stride of @math{3} and a horizontal stride of @math{2} to produce a @math{2 \times 4} 
 output image.
 
 @inlinecode
    _____                     _____                      _____                    _____   
   |O O O|P P P P P P     P P|O O O|P P P P      P P P P|O O O|P P    P P P P P P|O O O|  
   |O O O|X X X X X P     P X|O O O|X X X P      P X X X|O O O|X P    P X X X X X|O O O|
   |O_O_O|X X X X X P     P X|O_O_O|X X X P      P X X X|O_O_O|X P    P X X X X X|O_O_O|
    P X X X X X X X P     P X X X X X X X P      P X X X X X X X P    P X X X X X X X P
    P X X X X X X X P     P X X X X X X X P      P X X X X X X X P    P X X X X X X X P
    P X X X X X X X P     P X X X X X X X P      P X X X X X X X P    P X X X X X X X P
                                                                                            
        Y _ _ _               Y Y _ _                Y Y Y _             Y Y Y Y
        _ _ _ _               _ _ _ _                _ _ _ _             _ _ _ _
                                                                                          
                                                                                                
    P P P P P P P P P     P P P P P P P P P      P P P P P P P P P    P P P P P P P P P 
    P X X X X X X X P     P X X X X X X X P      P X X X X X X X P    P X X X X X X X P
    P_X_X X X X X X P     P X X_X_X X X X P      P X X X X_X_X X P    P X X X X X X_X_P
   |O O O|X X X X X P     P X|O O O|X X X P      P X X X|O O O|X P    P X X X X X|O O O| 
   |O O O|X X X X X P     P X|O O O|X X X P      P X X X|O O O|X P    P X X X X X|O O O| 
   |O_O_O|X X X X X P     P X|O_O_O|X X X P      P X X X|O_O_O|X P    P X X X X X|O_O_O| 
                                                                                            
        Y Y Y Y               Y Y Y Y                Y Y Y Y             Y Y Y Y
        Y _ _ _               Y Y _ _                Y Y Y _             Y Y Y Y  
  
  

 @endinlinecode
 
 The input, output, (implied) padding and window pixels are represented by `X`, `Y`, `P` 
 and `O` respectively.
  
  
 @note For simplicity, the input and output channel depths are not shown (or equivalently, are presumed to be 1) in 
        the diagram above.


### Configuration Options

The following sections describe configurable options for the @oper{conv2d_deep} operator. Configuration options can
be set by adding the appropriate project-wide build flag.

#### CONFIG_SYMMETRIC_SATURATION_conv2d_deep

By default, when the @oper{conv2d_deep} operator applies saturation logic to its output, it uses the standard bounds 8-bit 
signed integers, namely `(-128, 127)`. However, the XS3 VPU hardware is designed to apply symmetric saturation bounds of
`(-127,127)`. Using the asymmetric bounds for this operation is more expensive, but necessary for some applications.

If either the preprocessor symbol `CONFIG_SYMMETRIC_SATURATION_conv2d_deep` is defined to `1`, or if 
`CONFIG_SYMMETRIC_SATURATION_conv2d_deep` is undefined, but `CONFIG_SYMMETRIC_SATURATION_GLOBAL` is defined to `1`,
then the symmetric saturation bounds will be used instead. 

Note that this option *only* affects the output saturation bounds. It does *not* affect the saturation bounds of the
intermediate 32-bit or 16-bit accumulators.

See nn_config.h.

