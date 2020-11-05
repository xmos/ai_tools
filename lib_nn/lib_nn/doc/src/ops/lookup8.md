

# lookup8 #                                     {#oper_lookup8}


### Description 

This operator uses a look-up table using each element of the input vector @tensor{x} as an index to get each element 
of the output vector @tensor{y}.


### Parameters 

#### Hyperparameters        {#lookup8_hyperparams}

The following are the hyperparameters of @oper{lookup8}.

@par

<table>
<tr><th>Symbol(s)       <th>Description

<tr><td>@tensor_shape{N}            <td>The length of the input vector (in elements).
</table>

##### Hyperparameter Constraints

* None

#### Data Parameters

The following are input and output parameters of @oper{lookup8}.

@par

<table>

<tr><th colspan="2">Symbol          <th>Direction   <th>Shape               <th>Description

<tr><td colspan="2">@math{y}        <td>out         <td><i>scalar</i>       <td>The output index.
<tr><td colspan="2">@tensor{x}      <td>in          <td>@math{(N)}          <td>The input vector.
<tr><td colspan="2">@tensor{T}      <td>in          <td>@tensor_shape{256}  <td>The look-up table.

</table>


### Operation Performed

 
@f[
     y\left[k\right] = T\left[x\left[k\right]\right] \text{for } 0 \leq k \lt N
@f]
 

where the parameters are as described above.

### Example Diagram

@todo Create diagram
