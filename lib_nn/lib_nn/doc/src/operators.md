

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
layers of convolutions and pooling. Each occurrence of an operator within the network has a set of bound (and unchanging)
hyperparameters which describe the structure of the work to be performed, such as the size of a convolution window, or the 
number of channels processed by a pooling operation. An operator together with its bound hyperparameters constitutes a
concrete instance of that operator. An operator instance is represented in `lib_nn` by a _plan_ (together with one or more 
jobs).

#### Jobs

While a plan encapsulates the work to be done by the operator instance, it is often beneficial to split the actual execution
of the work into multiple parts. This may be done, for example, to reduce latency by dividing the work among multiple cores 
that can run in parallel, or to reduce the memory overhead by only keeping part of the parameters or data in S-RAM at a
time. Each block of work to be performed is called a _job_. In `lib_nn`, each job corresponds to a subset of the data to be
output by an operator instance. In some operators a job will compute a rectangular subset of an output image, while in others
a job will compute a contiguous block of the output's memory.

Plans and jobs are initialized together (usually during the initial boot sequence), and are later used when invoking an
instance of an operator.

@note * Several operators brazenly flout the *plan* / *job* convention. A future update may correct this.

@note * While conceptually plans represent the whole of the work to be performed, in practice, the information required is
      usually split between the struct representing the plan and structs representing the jobs.


## Other Remarks

#### Logical vs API Entities

You will find that in both the operator documentation as well as the API documentation, a distinction is made (or implied)
between conceptual ("logical") objects and their representations or corrolaries in the API. For example, in the documentation 
for maxpool2d(), there are references to both @tensor{X} and `X`. An attempt has been made 

## Operator Table       {#lib_nn_operator_table}

The following table enumerates the custom operators provided by this library.

@par


<table>
<caption id="multi_row">`lib_nn` Operators</caption>

<tr><th>Operator    <th colspan="2">Functions               <th colspan="2">Structs     <th> X
<tr><th>            <th> Job Invocation  <th> Initialization    <th> Plan       <th>Job     <th>

<tr><td>@oper{conv2d_deep}          <td>conv2d_deep()           <td>conv2d_deep_init()
                                    <td>nn_conv2d_deep_plan_t   <td>nn_conv2d_deep_job_t
                                    <td>

<tr><td>@oper{conv2d_shallowin}     <td>conv2d_shallowin()          <td>conv2d_shallowin_init()
                                    <td>nn_conv2d_shallowin_plan_t  <td>nn_conv2d_shallowin_job_t
                                    <td>

<tr><td>@oper{conv2d_1x1}           <td>conv2d_1x1()                <td>conv2d_1x1_init()
                                    <td>nn_conv2d_1x1_plan_t        <td>nn_conv2d_1x1_job_t
                                    <td>

<tr><td>@oper{conv2d_depthwise}     <td>conv2d_depthwise()          <td>conv2d_depthwise_init()
                                    <td>nn_conv2d_depthwise_plan_t  <td>nn_conv2d_depthwise_job_t
                                    <td>

<tr><td>@oper{maxpool2d}            <td>maxpool2d()                 <td>maxpool2d_init()
                                    <td>nn_maxpool2d_plan_t         <td>nn_pool2d_job_t
                                    <td>

<tr><td>@oper{avgpool2d}            <td>avgpool2d()                 <td>avgpool2d_init()
                                    <td>nn_avgpool2d_plan_t         <td>nn_pool2d_job_t
                                    <td>

<tr><td>@oper{avgpool2d_global}     <td>avgpool2d_global()          <td>avgpool2d_global_init()
                                    <td>nn_avgpool2d_global_plan_t  <td>nn_avgpool2d_global_job_t
                                    <td>

<tr><td>@oper{fully_connected_16}   <td>fully_connected_16()        <td>fully_connected_init()
                                    <td>nn_fully_connected_plan_t   <td>nn_fully_connected_job_t
                                    <td>

<tr><td>@oper{argmax_16}            <td>argmax_16()                 <td align="center">N/A
                                    <td align="center">N/A          <td align="center">N/A
                                    <td>

<tr><td>@oper{requantize_16_to_8}   <td>requantize_16_to_8()        <td>requantize_16_to_8_init()
                                    <td align="center">N/A          <td>nn_requantize_16_to_8_job_t
                                    <td>

<tr><td>@oper{lookup8}              <td>lookup8()                   <td align="center">N/A
                                    <td align="center">N/A          <td align="center">N/A
                                    <td>



</table>



