These are 5 example models; in order of complexity

* ``app_no_flash`` a single model, no flash memory used. This is the
  fastest but most pressure on internal memory.

* ``app_flash_single_model`` a single model, with learned parameters in
  flash memory. This removes a lot of pressure on internal memory

* ``app_flash_two_models`` two models, with learned parameters in flash memory.

* ``app_flash_two_models_one_arena`` two models, with learned parameters in
  flash memory. The models share a single arena (scratch memory).

