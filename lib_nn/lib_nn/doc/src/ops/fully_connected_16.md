

# fully_connected_16 #                                     {#oper_fully_connected_16}


### Description 

This operator performs a matrix-vector multiplication with an additional scale and bias. The input matrix @tensor{W} and vector 
@tensor{x} are each 8 bits deep and the output vector @tensor{y} is 16 bits deep.

The @oper{fully_connected_16} operator requires a plan and one or more jobs to be initialized before it can be invoked. See 
@ref fully_connected_16_api below.

To reduce the bit-depth of the output vector @tensor{y}, this operator can be followed by @oper_ref{requantize_16_to_8}.

### Parameters 

#### Hyperparameters        {#fully_connected_16_hyperparams}

The following are the hyperparameters of @oper{fully_connected_16}. The hyperparameters for an instance of an operator are fixed 
at initialization. Instances of the @oper{fully_connected_16} operator that share the same hyperparameters may also share the same 
plan and jobs.

@par

<table>
<tr><th>Symbol(s)       <th>Description

<tr><td>@tensor_shape{N}            <td>The length of the input vector (in elements).
<tr><td>@tensor_shape{M}            <td>The length of the output vector (in elements).
</table>

##### Hyperparameter Constraints

* The input length must be a multiple of `4`.
  * @math{ N = 0 \left( \text{mod } 4 \right) }

#### Data Parameters

The following are input and output parameters of @oper{fully_connected_16}. These parameters are supplied only when the job invocation occurs,
and may change from invocation to invocation.

@par

<table>
<tr><th colspan="2">Symbol          <th>Direction   <th>Shape               <th>Description

<tr><td colspan="2">@tensor{y}      <td>out         <td>@math{(M)}          <td>The output vector.
<tr><td colspan="2">@tensor{x}      <td>in          <td>@math{(N)}          <td>The input vector.
<tr><td colspan="2">@tensor{W}      <td>in          <td>@math{(M,N)}        <td>The weight matrix.
<tr><td colspan="2">[`BSO`]         <td>in          <td>                    <td>The elements of the bias-scale-offset array (see @ref out_shift_scale).
<tr><td>        <td>@tensor{b}      <td>            <td>@math{M}            <td>The output biases.        
<tr><td>        <td>@tensor{s_1}    <td>            <td>@math{M}            <td>The first output shifts.
<tr><td>        <td>@tensor{s_2}    <td>            <td>@math{M}            <td>The output scales.
<tr><td>        <td>@tensor{o_a}    <td>            <td>@math{M}            <td>The output offset scales.
<tr><td>        <td>@tensor{o_b}    <td>            <td>@math{M}            <td>The output offset values.
<tr><td>        <td>@tensor{s_3}    <td>            <td>@math{M}            <td>The final output shifts.
</table>



### Operation Performed

@f[

     v\left[p \right] = b_i + \sum_{r=0}^{N-1} \left( W[p,r] \cdot x[r] \right)\\\
  \\\  
     y\left[p \right] = sat_{16}\left(\frac{\left(sat_{16}\left(\frac{v\left[p \right]}
             {2^{s_{1p}}}\right)\cdot s_{2p}\right)}{2^{s_{3p}}}\right) \text{for } 0 \leq p \lt C_{out}

@f]


where

@par
@tensor{v} is an intermediate vector (holding the 32-bit accumulators),

@par
@math{p} is the output index, and

@par
the remaining parameters are as described above.

### Example Diagram

@todo Create diagram


### API                     {#fully_connected_16_api}

Invoking an instance of @oper{fully_connected_16} is done with a call to fully_connected_16(). fully_connected_16() takes a pointer to an initialized plan 
(instance of `nn_fully_connected_plan_t`) and an initialized job (instance of `nn_fully_connected_job_t`). Initialization is done with a call
to fully_connected_init().

Each call to fully_connected_16() will execute exactly one job. A @oper{fully_connected_16} job computes a contiguous subset of 
the output vector's elements (which can be the entire vector if only one job is desired). For each job the user indicates a 
starting channel as well as the number of channels to be computed by that job. See fully_connected_init() for more details 
(and constraints).

It is the user's responsibility to ensure that all initialized jobs collectively compute the entire output vector (no gaps) and do not
compute outputs redundantly (overlapping jobs).

If a network uses multiple instances of the @oper{fully_connected_16} operator, they may share the structs representing the plan and any jobs 
*if and only if* the instances share identical hyperparameters (see @ref fully_connected_16_hyperparams).

