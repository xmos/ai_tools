

# avgpool2d_global                                     {#oper_avgpool2d_global}


### Description 

This operator performs a 2D global average pooling operation. A global average pooling operation is similar to average 
pooling when the pooling window covers the entire input image, with an additional bias term included (see 
@ref avgpool2d_global_op below). 
The result is an 8-bit vector in which the index corresponds to input channel.

### Parameters 

#### Hyperparameters        {#avgpool2d_global_hyperparams}

The following are the hyperparameters of @oper{avgpool2d_global}.

@par

<table>
<tr><th>Symbol(s)       <th>Description

<tr><td>@tensor_shape{X_h, X_w}    <td>The input image height and width, respectively.
<tr><td>@math{X_c}                 <td>The channel count of both the input image and output vector.
</table>

##### Hyperparameter Constraints

* The input (and output) channel count must be a multiple of `4`.
  * @math{ X_c = 0 \left( \text{mod } 4 \right) }

#### Data Parameters

The following are input and output parameters of @oper{avgpool2d_global}.

@par

<table>
<tr><th colspan="2">Symbol      <th>API Arg     <th>Direction   <th>Bit-depth   <th>Shape                       <th>Description
<tr><td colspan="2">@tensor{y}  <td>`Y`         <td>out         <td>8           <td>@math{(X_c)}                
<td>The output vector.

<tr><td colspan="2">@tensor{X}  <td>`X`         <td>in          <td>8           <td>@math{(X_h, X_w, X_c)}      
<td>The input image.

<tr><td colspan="2">@math{b}    <td>`bias`      <td>in          <td>32          <td><i>scalar</i></a>               
<td>The bias with which accumulators are initialized.

<tr><td colspan="2">@math{s}    <td>`scale`     <td>in          <td>8           <td><i>scalar</i></a>               
<td>Each input pixel value is multiplied by this during accumulation.

<tr><td colspan="2">@math{r}    <td>`shift`     <td>in          <td>16          <td><i>scalar</i></a>               
<td>The right-shift applied to accumulators to produce a final result.

</table>



### Operation Performed

@f[

     y\left[p \right] =  \left(b + \sum_{r=0}^{X_h-1}\sum_{c=0}^{X_w-1}
                        \left(s \cdot X\left[r,c,p \right]\right)\right)
                        \cdot 2^{-r}  \\\

@f]

where  

@par
@math{p} is the output channel index, and
@par
the remaining parameters are as described above.

#### Accumulator Saturation

The internal accumulators are signed 32-bit integers which saturates at symmetric 32-bit bounds. 
On a sufficiently large input image it is possible for an accumulator to (silently) saturate.
Without regard to the actual distribution of input pixel values, the accumulator can always accomodate _at least_ 
@math{ \frac{2^{31}-b}{2^{7}\cdot |s|}-1 } pixels. With @math{b=0} and @math{s=127}, that is approximately
@math{2^{17}} pixels, roughly  the number of pixels in a @math{362} by @math{362} image.

In practice it will likely accomodate many times that, particularly if the input pixels are expected to be 
near-zero mean.

#### Choosing a Scale and Shift

For simplicity, the following assumes @math{b = 0}.

The scale @math{s} and the shift @math{r} together determine the overall scaling of the result. The raw (unscaled, 
unbiased) sum of pixel values for a channel is ultimately multiplied by a factor of @math{\alpha} where

@f[
    \alpha = s\cdot2^{-r}
@f]

The pair then effectively represent a floating-point value with an 8-bit mantissa and a strictly non-positive exponent.

While @oper{avgpool2d_global} nominally computes the mean pixel value of each channel, it need not be the case that
@math{\alpha = \frac{1}{X_h \cdot X_w}}. This is useful because in practice computing a strict mean will tend to 
compress the range of outputs relative to inputs, which may represent a loss of available information. Computing a 
scaled mean can avoid such unnecessary loss.

While determining the most appropriate scale factor is case-specific and outside the scope of this documentation, once
an ideal value, @math{\hat{\alpha}}, is chosen, the following should be used to maximize the precision of the result.

@f[

    c = floor(log_2(\hat{\alpha}))    \\\
    \hat{s} = round\left( 2^6\cdot \frac{\hat{\alpha}}{2^c}  \right)   \\\
    \hat{r} = 6 - c    \\\
    s = \begin{cases}1 &  c = log_2(\hat\alpha)\\ \hat{s} & otherwise\end{cases}    \\\
    r = \begin{cases}-c &  c = log_2(\hat\alpha)\\ \hat{r} & otherwise\end{cases}    \\\

@f]

When @math{\hat\alpha} is not an exact power of @math{2}, this ensures that @math{64 \lt s \le 127}, making optimal use
of the available bits. When @math{\hat\alpha} is an exact (non-positive) power of 2, a simple right-shift is optimal,
and so this maximally reduces risk of 32-bit accumulator saturation.

### Example Diagram

@todo Create diagram

### Configuration Options

The following sections describe configurable options for the @oper{avgpool2d_global} operator. Configuration options can
be set by adding the appropriate project-wide build flag.

#### CONFIG_SYMMETRIC_SATURATION_avgpool2d_global

By default, when the @oper{avgpool2d_global} operator applies saturation logic to its output, it uses the standard bounds 8-bit 
signed integers, namely `(-128, 127)`. However, the XS3 VPU hardware is designed to apply symmetric saturation bounds of
`(-127,127)`. Using the asymmetric bounds for this operation is more expensive, but necessary for some applications.

If either the preprocessor symbol `CONFIG_SYMMETRIC_SATURATION_avgpool2d_global` is defined to `1`, or if 
`CONFIG_SYMMETRIC_SATURATION_avgpool2d_global` is undefined, but `CONFIG_SYMMETRIC_SATURATION_GLOBAL` is defined to `1`,
then the symmetric saturation bounds will be used instead. 

Note that this option *only* affects the output saturation bounds. It does *not* affect the saturation bounds of the
intermediate 32-bit or 16-bit accumulators.

See nn_config.h.

