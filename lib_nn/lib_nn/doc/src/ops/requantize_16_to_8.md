

# requantize_16_to_8 #                                     {#oper_requantize_16_to_8}


### Description 

This operator consumes a 16-bit input vector @tensor{x} to produce an 8-bit output vector @tensor{y}. Conceptually, if the contents
of the 16-bit input vector are considered to be a signed Q0.15 fixed-point scalar, then this operation simply changes each element 
to a signed Q0.7 fixed-point scalar, modifying the precision, rather than the value. Alternatively, if the input and output vectors 
are considered to be signed integers, then the operation divides each input element by @math{2^{8}}, rounding the result to the 
nearest integer. These interpretations are equivalent.

### Parameters 

#### Hyperparameters        {#requantize_16_to_8_hyperparams}

The following are the hyperparameters of @oper{requantize_16_to_8}. The hyperparameters for an instance of an operator are fixed at 
initialization. Instances of the @oper{requantize_16_to_8} operator that share the same hyperparameters may also share the same jobs.

@par

<table>
<tr><th>Symbol(s)       <th>Description

<tr><td>@tensor_shape{N}            <td>The length of the input vector (in elements).
</table>

##### Hyperparameter Constraints

* None

#### Data Parameters

The following are input and output parameters of @oper{requantize_16_to_8}.

@par

<table>

<tr><th colspan="2">Symbol          <th>Direction   <th>Shape               <th>Description

<tr><td colspan="2">@tensor{y}      <td>out         <td>@math{(N)}          <td>The output index.
<tr><td colspan="2">@tensor{x}      <td>in          <td>@math{(N)}          <td>The input vector.

</table>


### Operation Performed

@f[
     y\left[k\right] \overset{8-bit}{\longleftarrow} x\left[k\right] \text{for } 0 \leq k \lt N
@f]

where the parameters are as described above.

### Example Diagram

@todo Create diagram
### Configuration Options

The following sections describe configurable options for the @oper{requantize_16_to_8} operator. Configuration options can
be set by adding the appropriate project-wide build flag.

#### CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8

By default, when the @oper{requantize_16_to_8} operator applies saturation logic to its output, it uses the standard bounds 8-bit 
signed integers, namely `(-128, 127)`. However, the XS3 VPU hardware is designed to apply symmetric saturation bounds of
`(-127,127)`. Using the asymmetric bounds for this operation is more expensive, but necessary for some applications.

If either the preprocessor symbol `CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8` is defined to `1`, or if 
`CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8` is undefined, but `CONFIG_SYMMETRIC_SATURATION_GLOBAL` is defined to `1`,
then the symmetric saturation bounds will be used instead. 

Note that this option *only* affects the output saturation bounds. It does *not* affect the saturation bounds of the
intermediate 32-bit or 16-bit accumulators.

See nn_config.h.

