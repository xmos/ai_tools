install:
	mkdir -p build
	(cd build && cmake .. && make install -j4)

clean:
	rm -rf build
	rm -rf libs

test:
	(cd tests && make test)
