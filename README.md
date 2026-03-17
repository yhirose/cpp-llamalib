# cpp-llamalib

[![CI](https://github.com/yhirose/cpp-llamalib/actions/workflows/ci.yml/badge.svg)](https://github.com/yhirose/cpp-llamalib/actions/workflows/ci.yml)

A C++17 single-file header-only wrapper for [llama.cpp](https://github.com/ggml-org/llama.cpp).<br>
Just include **cpp-llamalib.h** to call llama.cpp with a simple, high-level API.

## Features

- Header-only — single file, no build step for the wrapper itself
- Simple `generate()` API with streaming support
- Thread-safe concurrent generation via slot pool
- Custom sampler configuration

## Integration

Please copy `cpp-llamalib.h` into your project and `#include` it.

Your project must link against llama.cpp (`llama` and `ggml` libraries).<br>
Tested against llama.cpp [b8389](https://github.com/ggml-org/llama.cpp/releases/tag/b8389), 2026-03-17.

## Examples

### Text generation

Load a GGUF model and generate text from a prompt. The result is returned as a `std::string`.

```cpp
#include "cpp-llamalib.h"

llamalib::Llama llm("model.gguf");
auto result = llm.generate("Explain C++ RAII in one sentence.");
```

### Streaming

Pass a callback to receive tokens as they are generated. Return `false` from the callback to stop early.

```cpp
llm.generate("Write a haiku about the sea.", [](const std::string &token) {
    std::cout << token << std::flush;
    return true;  // return false to stop early
});
```

### Custom parameters

Configure context size, token limits, temperature, and concurrency via `Options`.

```cpp
llamalib::Options opts;
opts.n_ctx = 4096;
opts.max_tokens = 256;
opts.temperature = 0.7f;
opts.n_slots = 4;  // concurrent generation slots

llamalib::Llama llm("model.gguf", opts);
```

### Error handling

All errors are reported as `std::runtime_error` — invalid model paths, tokenization failures, and context overflow.

```cpp
try {
    llamalib::Llama llm("/bad/path.gguf");
} catch (const std::runtime_error &e) {
    // "Failed to load model"
}

try {
    // Throws if prompt + max_tokens exceeds context size
    auto result = llm.generate(very_long_prompt);
} catch (const std::runtime_error &e) {
    // "Prompt too long: 1024 prompt tokens + 512 max_tokens exceeds context size 1024"
}
```

### Concurrent generation

Set `n_slots` to enable multiple concurrent generations. Calls to `generate()` are thread-safe — if all slots are busy, callers block until a slot becomes available.

```cpp
llamalib::Options opts;
opts.n_slots = 2;
llamalib::Llama llm("model.gguf", opts);

// 4 requests on 2 slots: 2 run immediately, 2 wait for a free slot
std::vector<std::future<std::string>> futures;
for (int i = 0; i < 4; i++) {
    futures.push_back(std::async(std::launch::async, [&llm] {
        return llm.generate("Hello");
    }));
}
```

### Custom sampler

Override the default sampler chain by providing a `sampler_config` callback. You have full access to llama.cpp's sampler API.

```cpp
llamalib::Options opts;
opts.sampler_config = [](llama_sampler *chain) {
    llama_sampler_chain_add(chain, llama_sampler_init_temp(0.0f));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(42));
};

llamalib::Llama llm("model.gguf", opts);
```

## API Reference

### `llamalib::Options`

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `n_gpu_layers` | `int` | `99` | Number of layers to offload to GPU |
| `n_ctx` | `int` | `2048` | Context window size |
| `temperature` | `float` | `0.3f` | Sampling temperature |
| `max_tokens` | `int` | `512` | Default max tokens to generate |
| `n_slots` | `int` | `1` | Number of concurrent generation slots |
| `sampler_config` | `SamplerConfig` | `nullptr` | Custom sampler chain configuration callback |

### `llamalib::Llama`

#### Constructor

```cpp
Llama(const std::string &model_path, const Options &params = {})
```

Loads the model and creates context/sampler slots. Throws `std::runtime_error` on failure.

#### `generate`

```cpp
std::string generate(const std::string &prompt)
std::string generate(const std::string &prompt, int max_tokens)
std::string generate(const std::string &prompt, const StreamCallback &callback)
std::string generate(const std::string &prompt, int max_tokens, const StreamCallback &callback)
```

Generates text from a prompt. Thread-safe — concurrent calls queue for available slots.

- `max_tokens` — overrides `Options::max_tokens` for this call
- `callback` — called with each token as it is generated; return `false` to stop early

Throws `std::runtime_error` if the prompt is too long for the context window or if decoding fails.

### Type aliases

```cpp
using StreamCallback = std::function<bool(const std::string &token)>;
using SamplerConfig = std::function<void(llama_sampler *chain)>;
```

## License

MIT
