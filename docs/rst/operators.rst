Operators
=========

Virtually all tensorflow-lite-for-micro operators are supported
with the exception of Variables, While, and If. Only very few operators
have been optimised to run efficiently on XCORE; those that typically
account for 99% of the execution time. 

Optimized operators
-------------------

The following operators can be optimized by the xcore optimizer into an
equivalent faster or more memory efficient operator:

* Conv2D

* Conv2DDepthwise

* Conv2DTranspose

* FullyConnected

* MaxPool2D

* AvgPool2D

* Add

* Mul

* Concatenate

* Pad

* Slice or StridedSlice with a stride of 1

* Tanh, Sigmoid, Hardswish, Relu

We are always interested to know of operators that take on a large
proportion of your model.

Constraints on operators
------------------------

Some xcore optimizable operators have constraints on them which dictate
situations where they can not be optimized. In particular:

* Make sure that each convolution outputs a multiple of FOUR channels.

* For optimal speed, the number of input channels should be a multiple of
  16, otherwise 4.

* For a first image convolution that typically has three channels (YUV,
  RGB), the graph transformer will insert a fast pad from three to four.

* For a convolution, execution is fast when the bias term is reasonably
  close to zero.

