

# lookup8 #                                     {#oper_lookup8}


### Description 

This operator uses a look-up table using each element of the input vector @tensor{x} as an index to get each element 
of the output vector @tensor{y}.

Unlike most other operators, the @oper{lookup8} operator requires no plan or jobs to be initialized before it can 
be invoked. See @ref lookup8_api below.


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

### API                     {#lookup8_api}

Invoking an instance of @oper{lookup8} is done with a call to lookup8(). No plan or jobs are required and no initialization
is necessary.

To apply the @oper{lookup8} operation on a multi-dimensional tensor, simply supply the product of the tensors dimensions as 
@math{N}.

