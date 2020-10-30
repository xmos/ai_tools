# Running unit tests with clang address sanitizer

Make sure clang in installed.  Then build with address sanitizer enabled:

    > make PLATFORM=x86 SANITIZE=true

The `SANITIZE=true` option sets CC=clang and sets some compiler flags to emable address sanitization.  

    -fsanitize=address
    -fsanitize-recover=address

Run the application with the following command:

    > ./bin/x86/unit_test

The command above will halt on the first asan error.  Run with `halt_on_error=false` to prevent the application from halting.

    > ASAN_OPTIONS=halt_on_error=false ./bin/x86/unit_test

