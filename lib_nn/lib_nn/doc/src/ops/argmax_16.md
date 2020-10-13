

# argmax_16 #                                     {#oper_argmax_16}


### Description 

This operator executes an argument maximization (@math{argmax_k\\{x[k]\\}}) function, which returns the index @math{k} 
of the maximum element of the vector @tensor{x}. The function is applied to a 16-bit input vector @tensor{x} to get a 
returned integer @math{y}.

### Parameters 

#### Hyperparameters        {#argmax_16_hyperparams}

The following are the hyperparameters of @oper{argmax_16}.

@par

<table>
<tr><th>Symbol(s)       <th>Description

<tr><td>@tensor_shape{N}            <td>The length of the input vector (in elements).
</table>

##### Hyperparameter Constraints

* None

#### Data Parameters

The following are input and output parameters of @oper{argmax_16}.

@par

<table>

<tr><th colspan="2">Symbol          <th>Direction   <th>Shape               <th>Description

<tr><td colspan="2">@math{y}        <td>out         <td><i>scalar</i>       <td>The output index.
<tr><td colspan="2">@tensor{x}      <td>in          <td>@math{(N)}          <td>The input vector.

</table>


### Operation Performed

 
 @f[
      y \leftarrow argmax_{k}\{ x\left[k\right] \} \text{ for } 0 \leq k \lt N
 @f]
 

where the parameters are as described above.

### Example Diagram

For example, if the input vector were the following 16-bit array:

@code
    int16_t input_vector[8] = { 0, -534, 1224, -32768, 16000, 8, 100, -32 };
@endcode

Then the result of an @oper{argmax_16} operation would be `4`, as `input_vector[4]` (value: `16000`) is the maximum element.
