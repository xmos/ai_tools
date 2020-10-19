

# maxpool2d #                                     {#oper_maxpool2d}


### Description 

This operator performs a 2D maximum pooling operation. A maximum pooling operation slides a window across an input image in two dimensions
and in each position of the pooling window it selects the maximum element for each channel within the window and sets the corresponding
output channels to those values for the corresponding output pixel.

This operator does not support any implied padding. All cells of the pooling window must be inside the input image for all output pixels.
If padding is needed, the image should be explicitly padded in memory prior to invoking this operator.

### Parameters 

#### Hyperparameters        {#maxpool2d_hyperparams}

The following are the hyperparameters of @oper{maxpool2d}.

@par

<table>
<tr><th>Symbol(s)       <th>Description

<tr><td>@tensor_shape{X_h, X_w}         <td>The input image height and width, respectively.
<tr><td>@tensor_shape{Y_h, Y_w}         <td>The output image height and width, respectively.
<tr><td>@math{X_c}                      <td>The channel count of both the input and output images.
<tr><td>@tensor_shape{K_h, K_w}         <td>The pooling window height and width, respectively.
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
 * @math{ W_{r0} + W_{vert} * \left( Y_h - 1 \right) + K_h  \lt X_h }
 * @math{ W_{c0} + W_{hori} * \left( Y_w - 1 \right) + K_w  \lt X_w }

#### Data Parameters

The following are input and output parameters of @oper{maxpool2d}.

@par

<table>
<tr><th colspan="2">Symbol          <th>Direction   <th>Shape                       <th>Description
<tr><td colspan="2">@tensor{Y}      <td>out         <td>@math{(Y_h, Y_w, X_c)}      <td>The output image.
<tr><td colspan="2">@tensor{X}      <td>in          <td>@math{(X_h, X_w, X_c)}      <td>The input image.
</table>



### Operation Performed

@f[
     
     V_{r,c}\left[u,v,p \right] = X\left[
                                     W_{r0} + r\cdot W_{vert} + u,
                                     W_{c0} + c\cdot W_{hori} + v,
                                     p \right] \\\
     \text{ for } 0 \leq u \lt W_{height} \text{ and } 0 \leq v \lt W_{width} \\\
  \\\  
     Y\left[r,c,p\right]= max\left( \bar V_{r,c} \right)

@f]

where  

@par
@tensor{V} is an intermediate tensor with shape @tensor_shape{K_h, K_w, X_c}, which contains only the portion of 
    @tensor{X} that is within the pooling window for output pixel @math(r,c).

@par
@math{(r,c,p)} are the output row, column and channel,

@par
@math{max\left(\cdot\right)} gives a vector containing the maximum element value within each channel-slice 
    of @tensor{V}, and
@par
the remaining parameters are as described above.

### Example Diagram

@todo Create diagram

