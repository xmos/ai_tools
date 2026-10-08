xmos ai_tools change log
========================

1.4.3.dev50
-----------

  * CHANGED: Size optimization is now enabled by default for VX4 runtime builds; ENABLE_SIZE_OPT is no longer required.
  * CHANGED: Linux CI builds run natively; the root Docker build image is removed.
  * CHANGED: Linux wheels now require glibc 2.31 or later, using the manylinux_2_31_x86_64 platform tag.
  * CHANGED: lib_nn built as a static library and linked accordingly.
  * CHANGED: Additional lib_nn improvements for elementwise operations, lookup, quantization and matrix multiplication, grouped as dependency improvements.
  * CHANGED: Updated examples to use the xcommon cmake build system.
  * REMOVED: The app_flash_4 and app_no_flash_with_ioserver examples.

1.4.3.dev40
-----------

  * ADDED: CMake option (`ENABLE_SIZE_OPT`) to optimize `libtflitemicro.a` size on Vx4.
  
1.4.3.dev39
-----------

  * ADDED: xcommon_cmake support
  * CHANGED: Replaced xmake with xcommon cmake for building the mobilenetv2 application.

1.4.3.dev37
-----------

  * FIXED: mul_elementwise and add_elementwise for vx4b platform (reference fallback for now).  

1.4.3.dev33
-----------

  * FIXED: stack-size warnings for vx4b platform.
