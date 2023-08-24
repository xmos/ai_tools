Example applications
----------------------------

These are 5 example models; in order of complexity

* `app_no_flash <app_no_flash/README.rst>`_  - a single model, no flash memory used. This is the
  fastest but most pressure on internal memory.

* `app_flash_single_model <app_flash_single_model/README.rst>`_ - a single model, with learned parameters in
  flash memory. This removes a lot of pressure on internal memory.

* `app_flash_two_models <.app_flash_two_models/README.rst>`_ - two models, with learned parameters in flash memory.

* `app_flash_two_models_one_arena <app_flash_two_models_one_arena/README.rst>`_ - two models, with learned parameters in
  flash memory. The models share a single tensor arena (scratch memory).

* `app_profiling <app_profiling/README.rst>`_ - demonstrates how to enable and use profiling to speed up execution.
