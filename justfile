build:
    cmake -B build
    cmake --build build -j$(sysctl -n hw.ncpu)

test: build
    ctest --test-dir build --output-on-failure
