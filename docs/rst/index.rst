AI tools User Guide
###################

This document gives an introduction how to use the XMOS ai tools and how to
deploy trained models on XCORE.AI. XCORE.AI is a multi-threaded
micro-controller by XMOS, capable of real-time IO, DSP, and fast execution
of neural networks.

We first explain which models are suitable to run on XCORE.AI, and what
configuration choices you have. After that we discuss how models are
executed on XCORE.AI and the flow that you can follow. Using flash to store
data is an important part of the tools, and that is covered in the last
main section.

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
