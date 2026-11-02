build:
    cmake -B build
    cmake --build build -j{{num_cpus()}}

test: build
    ctest --test-dir build --output-on-failure

update-llama-cpp:
    ./scripts/update-llama-cpp.sh
