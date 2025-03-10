AI tools User Guide
###################

This document gives an introduction how to use the XMOS ai-tools and how to
deploy trained models on XCORE.AI. XCORE.AI is a multi-threaded
micro-controller by XMOS, capable of real-time IO, DSP, and fast execution
of neural networks. The XMOS ai-tools take a (trained) neural network and
compile it to code that can be optimised and compiled to execute on
XCORE.AI. The optimisation and compilation process can be controlled in
order to make trade-offs between costs, speed, and model-size.

This document first explains which models are suitable to run on XCORE.AI,
and what configuration choices you have. After that we discuss how models
are executed on XCORE.AI. The third section on :ref:`work_flow` describes
how to download and run the tools. Tools are installed through ``pip`` and
can be executed either from the command-line or through Python. The tools
natively work with ``.tflite`` files, and there are two python-notebooks
that show how to go from ``pytorch`` instead. The final section of this
document describes how to use flash to store learned parameters.

Appendices cover the options that the ``xformer`` supports, the operators
that are optimised, and how to build the ``xformer`` from source should you
wish to do so.


.. include:: xcore-ai-coding.rst

|newpage|

.. include:: prepare.rst

|newpage|

.. include:: flow.rst

|newpage|

.. include:: pytorch.rst

|newpage|

.. include:: flash.rst

|newpage|

.. include:: changelog.rst

|appendices|

|newpage|

.. include:: options.rst

|newpage|

.. include:: operators.rst

|newpage|

.. include:: build-from-source.rst

|newpage|

.. include:: faq.rst
