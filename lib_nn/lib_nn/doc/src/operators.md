

# Operators


## Networks, Operators, Plans and Jobs

The current design of `lib_nn` centers around a concept hierarchy that breaks down as follows.

#### Networks

At the top level of the hierarchy is the concept of a _network_. A network is a sequence of operations and the data that
joins them that accomplishes some computational task, such as performing inference using a convolutional neural network.
`lib_nn` in its raw form (that is, without using the TensorFlow Lite Micro runtime) does not have any explicit semantic 
representation of a network, but a network is created by the sequence of invocations performed by a user of `lib_nn`.

#### Operators

Below the network is an _operator_. An operator is an abstraction representing a certain class of operations. For example,
@oper{avgpool2d} is an operator that performs 2D average pooling on images by sliding a pooling window of arbitrary size
in two dimensions around an input image to produce an output image. An operator is represented semantically in the
API by a set of struct definitions and functions capable of performing the necessary arithmetic.

#### Operator Instances

It will often be the case that a network must make use of the same operator multiple times, for example, by having alternating
layers of convolutions and pooling. Each occurrence of an operator within the network has a set of hyperparameters which 
describe the structure of the work to be performed, such as the size of a convolution window, or the 
number of channels processed by a pooling operation. An operator together with its hyperparameters constitutes a concrete 
instance of that operator.

#### Jobs

It is often beneficial to split the actual execution of the work for an operator instance into multiple parts. This may 
be done, for example, to reduce latency by dividing the work among multiple cores that can run in parallel, or to reduce 
the memory overhead by only keeping part of the parameters or data in SRAM at a time. Each block of work to be performed 
is referred to as a _job_. In `lib_nn`, each job corresponds to a subset of the data to be output by an operator 
instance. In some operators a job will compute a rectangular subset of an output image, while in others
a job will compute a contiguous block of the output's memory.


## Other Remarks

#### Logical vs API Entities

You will find that in both the operator documentation as well as the API documentation, a distinction is made (or implied)
between conceptual ("logical") objects and their representations or corrolaries in the API. For example, in the documentation 
for maxpool2d(), there are references to both @tensor{X} and `X`. In that case, @tensor{X} refers to a mathematical object,
whereas `X` is a pointer to a buffer containing a particular encoding of that same object. In general, this documentation 
attempts to use a convention whereby logical entities are identified using math script (e.g. @math{K}), and API or software
entities are indentified using a fixed-width font (e.g. `K`).

The distinction is particularly relevant in cases where certain optimizations require particular encodings. For example, most
of the operators require a set of vectors as inputs which are the biases and scaling parameters for the output channels. 
Logically those entities may be thought of as vectors (e.g. the bias vector @math{\bar B} and the scale vector @math{\bar s_2}), 
but the architecture requires an encoding in which these vectors are "boggled" together and padded out (hence the `BSO` arrays
of `nn_bso_block_t` structs).

## Operator Table       {#lib_nn_operator_table}

The following table enumerates the custom operators provided by this library.

@par


<table>
<caption id="multi_row">`lib_nn` Operators</caption>

<tr><th>Operator    <th colspan="2">Invocation Function(s)

<tr><td align="center" colspan="3">Convolution Operators
<tr><td>@oper_ref{conv2d_deep}          <td>conv2d_deep() 
                                        <br>conv2d_deep_ext()

<tr><td>@oper_ref{conv2d_shallowin}     <td>conv2d_shallowin()
                                        <br>conv2d_shallowin_ext()

<tr><td>@oper_ref{conv2d_1x1}           <td>conv2d_1x1()
                                        <br>conv2d_1x1_ext()

<tr><td>@oper_ref{conv2d_depthwise}     <td>conv2d_depthwise()
                                        <br>conv2d_depthwise_ext()

<tr><td align="center" colspan="6">Pooling Operators
<tr><td>@oper_ref{maxpool2d}            <td>maxpool2d()
                                        <br>maxpool2d_ext()            

<tr><td>@oper_ref{avgpool2d}            <td>avgpool2d()
                                        <br>avgpool2d_ext() 

<tr><td>@oper_ref{avgpool2d_global}     <td>avgpool2d_global()
                                        <br>avgpool2d_global_ext() 

<tr><td align="center" colspan="6">Miscellaneous Operators
<tr><td>@oper_ref{add_elementwise}      <td>add_elementwise()

<tr><td>@oper_ref{fully_connected_16}   <td>fully_connected_16()

<tr><td>@oper_ref{fully_connected_8}    <td>fully_connected_8()

<tr><td>@oper_ref{argmax_16}            <td>argmax_16()

<tr><td>@oper_ref{requantize_16_to_8}   <td>requantize_16_to_8()

<tr><td>@oper_ref{lookup8}              <td>lookup8()



</table>



