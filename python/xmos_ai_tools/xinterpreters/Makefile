.PHONY: install clean test

JOBS := $(shell nproc --ignore=1)

install:
	cmake -B build
	make -j$(JOBS) -C build install

clean:
	rm -rf build
	rm -rf libs
