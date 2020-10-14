

# avgpool2d #                                     {#oper_avgpool2d}


### Description 

This operator performs a 2D average pooling operation. An average pooling operation slides a window across an input image in two dimensions
and in each position of the pooling window it compute the average for each channel slice within the window and sets the corresponding
output pixel to that average.

This operator does not support any implied padding. All cells of the pooling window must be inside the input image for all output pixels.
If padding is needed, the image should be explicitly padded in memory prior to invoking this operator.

### Parameters 

#### Hyperparameters        {#avgpool2d_hyperparams}

The following are the hyperparameters of @oper{avgpool2d}.

@par

<table>
<tr><th>Symbol(s)       <th>Description

<tr><td>@tensor_shape{X_h, X_w}         <td>The input image height and width, respectively.
<tr><td>@tensor_shape{Y_h, Y_w}         <td>The output image height and width, respectively.
<tr><td>@math{X_c}                      <td>The channel count of both the input and output images.
<tr><td>@tensor_shape{W_h, W_w}         <td>The pooling window height and width, respectively.
<tr><td>@math{(W_{r0}, W_{c0})}         <td>The pooling window's start position, which are the coordinates of the upper-left pixel of the 
                                            pooling window (in the input image's coordinate space) corresponding to the output pixel at
                                            @math{(0,0)} (in the output image's coordinate space).
<tr><td>@math{(W_{vert}, W_{hori})}     <td>The convolution window's vertical and horizontal strides, respectively.
</table>

##### Hyperparameter Constraints

* The input (and output) channel count must be a multiple of `4`.
  * @math{ X_c = 0 \left( \text{mod } 4 \right) }
* The pooling window must never extend beyond the bounds of the input image.
 * @math{ W_{r0} \geq 0 }
 * @math{ W_{c0} \geq 0 }
 * @math{ W_{r0} + W_{vert} * \left( Y_h - 1 \right) + W_h  \lt X_h }
 * @math{ W_{c0} + W_{hori} * \left( Y_w - 1 \right) + W_w  \lt X_w }

#### Data Parameters

The following are input and output parameters of @oper{avgpool2d}.

@par

<table>
<tr><th colspan="2">Symbol          <th>Direction   <th>Shape                       <th>Description
<tr><td colspan="2">@tensor{Y}      <td>out         <td>@math{(Y_h, Y_w, X_c)}      <td>The output image.
<tr><td colspan="2">@tensor{X}      <td>in          <td>@math{(X_h, X_w, X_c)}      <td>The input image.
</table>



### Operation Performed

@f[
     
     Y\left[r,c,p \right] =  \frac{1}{W_h \cdot W_w} \cdot
                             \sum_{w_r=0}^{W_h-1}\sum_{w_c=0}^{W_w-1} 
                                 X\left[ W_{r0}+r\cdot W_{vert}+w_r,
                                 W_{c0}+c\cdot W_{hori}+w_c,
                                 p\right] \\\


@f]

where  

@par
@math{(r,c,p)} are the output row, column and channel, and
@par
the remaining parameters are as described above.

### Example Diagram

@todo Create diagram

### Configuration Options

The following sections describe configurable options for the @oper{avgpool2d} operator. Configuration options can
be set by adding the appropriate project-wide build flag.

#### CONFIG_SYMMETRIC_SATURATION_avgpool2d

By default, when the @oper{avgpool2d} operator applies saturation logic to its output, it uses the standard bounds 8-bit 
signed integers, namely `(-128, 127)`. However, the XS3 VPU hardware is designed to apply symmetric saturation bounds of
`(-127,127)`. Using the asymmetric bounds for this operation is more expensive, but necessary for some applications.

If either the preprocessor symbol `CONFIG_SYMMETRIC_SATURATION_avgpool2d` is defined to `1`, or if 
`CONFIG_SYMMETRIC_SATURATION_avgpool2d` is undefined, but `CONFIG_SYMMETRIC_SATURATION_GLOBAL` is defined to `1`,
then the symmetric saturation bounds will be used instead. 

Note that this option *only* affects the output saturation bounds. It does *not* affect the saturation bounds of the
intermediate 32-bit or 16-bit accumulators.

See nn_config.h.

