

# fully_connected_8 #                                     {#oper_fully_connected_8}


### Description 

This operator performs a matrix-vector multiplication with additional (per-output) scales and offsets (see 
@ref fully_connected_8_op below). The input matrix @tensor{W} and vector @tensor{x} are each 8 bits deep and the 
output vector @tensor{y} is 8 bits deep.

The @oper{fully_connected_8} operator requires no plan or jobs to be initialized before it can be invoked. Each call
is effectively a job.

For 16-bit outputs use the @oper_ref{fully_connected_16} operator instead.

### Parameters 

#### Hyperparameters        {#fully_connected_8_hyperparams}

The following are the hyperparameters of @oper{fully_connected_8}.

@par

<table>
<tr><th>Symbol(s)       <th>Description

<tr><td>@tensor_shape{N}            <td>The length of the input vector @tensor{x} (in elements).
<tr><td>@tensor_shape{M}            <td>The length of the output vector @tensor{y} (in elements).
</table>

##### Hyperparameter Constraints

* The input length must be a multiple of `4`. 
  * @math{ N = 0 \left( \text{mod } 4 \right) }

#### Data Parameters

The following are input and output parameters of @oper{fully_connected_8}.

@par

<table>
<tr><th colspan="2">Symbol          <th>Direction   <th>Bit-depth   <th>Shape               <th>Description

<tr><td colspan="2">@tensor{y}      <td>out         <td>8           <td>@math{(M)}          <td>The output vector.
<tr><td colspan="2">@tensor{x}      <td>in          <td>8           <td>@math{(N)}          <td>The input vector.
<tr><td colspan="2">@tensor{W}      <td>in          <td>8           <td>@math{(M,N)}        <td>The weight matrix.
<tr><td colspan="2">[`BSO`]         <td>in          <td>            <td><td>The elements of the bias-scale-offset array (see @ref out_shift_scale).
<tr><td>        <td>@tensor{b}      <td>            <td>32          <td>@math{M}            <td>The output biases.        
<tr><td>        <td>@tensor{s_1}    <td>            <td>16          <td>@math{M}            <td>The first output shifts.
<tr><td>        <td>@tensor{s_2}    <td>            <td>16          <td>@math{M}            <td>The output scales.
<tr><td>        <td>@tensor{o_a}    <td>            <td>16          <td>@math{M}            <td>The output offset scales.
<tr><td>        <td>@tensor{o_b}    <td>            <td>16          <td>@math{M}            <td>The output offset values.
<tr><td>        <td>@tensor{s_3}    <td>            <td>16          <td>@math{M}            <td>The final output shifts.
</table>



### Operation Performed         {#fully_connected_8_op}

@f[

     v\left[p \right] = b_i + \sum_{r=0}^{N-1} \left( W[p,r] \cdot x[r] \right)\\\
  \\\  
     y\left[p \right] = sat_{8}\left(\frac{\left(sat_{16}\left(\frac{v\left[p \right]}
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
