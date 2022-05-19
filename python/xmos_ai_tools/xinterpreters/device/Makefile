install:
	mkdir -p build
	(cd build && cmake .. && make install -j4)

clean:
	rm -rf build

test:
	(cd tests && make test)
