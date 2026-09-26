.PHONY: xinterpreters clean

xinterpreters:
	cmake -S . -B build
	cmake --build build --target install --parallel 8 --config Release

clean:
	rm -rf build
	rm -rf libs
