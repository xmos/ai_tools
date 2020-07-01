

# avgpool2d_global                                     {#oper_avgpool2d_global}


### Description 

This operator performs a 2D global average pooling operation. A global average pooling operation is similar to average pooling when
the pooling window covers the entire input image, with an additional bias term added in. For each channel slice all input pixel values 
are added together with the bias and the sum is divided by the number of input pixels. The result is a vector in which the index 
corresponds to input channel.

The @oper{avgpool2d_global} operator requires a plan and one or more jobs to be initialized before it can be invoked. See 
@ref avgpool2d_global_api below.

### Parameters 

#### Hyperparameters        {#avgpool2d_global_hyperparams}

The following are the hyperparameters of @oper{avgpool2d_global}. The hyperparameters for an instance of an operator are fixed at initialization.  
Instances of the @oper{avgpool2d_global} operator that share the same hyperparameters may also share the same plan and jobs.

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

The following are input and output parameters of @oper{avgpool2d_global}. These parameters are supplied only when the job invocation occurs,
and may change from invocation to invocation.

@par

<table>
<tr><th colspan="2">Symbol          <th>Direction   <th>Shape                       <th>Description
<tr><td colspan="2">@tensor{y}      <td>out         <td>@math{(X_c)}                <td>The output vector.
<tr><td colspan="2">@tensor{X}      <td>in          <td>@math{(X_h, X_w, X_c)}      <td>The input image.
<tr><td colspan="2">@math{b}        <td>in          <td><i>scalar</a>               <td>The bias.
</table>



### Operation Performed

@f[
     
     y\left[p \right] =  \frac{
                             b + \sum_{r=0}^{X_h-1}\sum_{c=0}^{X_w-1}\left( 
                                 X\left[r,c,p \right]\right)}{X_h\cdot X_w} \\\

@f]

where  

@par
@math{p} is the output channel index, and
@par
the remaining parameters are as described above.

### Example Diagram

@todo Create diagram

### API                     {#avgpool2d_global_api}

An instance of @oper{avgpool2d_global} is invoked with a call to avgpool2d_global(). avgpool2d_global() takes a pointer to an 
initialized plan (instance of `nn_avgpool2d_global_plan_t`) and an initialized job (instance of `nn_avgpool2d_global_job_t`). 
Initialization is done with a call to avgpool2d_global_init().

Each call to avgpool2d_global() will execute exactly one job. A @oper{avgpool2d_global} job computes a contiguous subset of 
the output vector's channels (which can be the entire vector if only one job is desired). For each job the user indicates a 
starting channel as well as the number of channels to be computed by that job. See avgpool2d_global_init() for more details 
(and constraints).

It is the user's responsibility to ensure that all initialized jobs collectively compute the entire output vector (no gaps) and 
do not compute outputs redundantly (overlapping jobs).

If a network uses multiple instances of the @oper{avgpool2d_global} operator, they may share the structs representing the plan 
and any jobs *if and only if* the instances share identical hyperparameters (see @ref avgpool2d_global_hyperparams).


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

