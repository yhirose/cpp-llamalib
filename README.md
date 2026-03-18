# cpp-llamalib

[![CI](https://github.com/yhirose/cpp-llamalib/actions/workflows/ci.yml/badge.svg)](https://github.com/yhirose/cpp-llamalib/actions/workflows/ci.yml)

A C++17 single-file header-only wrapper for [llama.cpp](https://github.com/ggml-org/llama.cpp).<br>
Just include **cpp-llamalib.h** to call llama.cpp with a simple, high-level API.

## Features

- Header-only — single file, no build step for the wrapper itself
- Two-tier API: raw `generate()` and template-aware `chat()`
- `ChatSession` wrapper for multi-turn conversations
- Thread-safe concurrent generation via slot pool
- Custom sampler configuration

## Integration

Please copy `cpp-llamalib.h` into your project and `#include` it.

Your project must link against llama.cpp (`llama` and `ggml` libraries).<br>
Tested against llama.cpp [b8389](https://github.com/ggml-org/llama.cpp/releases/tag/b8389), 2026-03-17.

## Examples

### Raw text generation

`generate()` sends the prompt directly to the model without any chat template formatting.

```cpp
#include "cpp-llamalib.h"

llamalib::Llama llm("model.gguf");
auto result = llm.generate("Once upon a time");
```

### Chat

`chat()` applies the model's embedded chat template before generation.

```cpp
// Single message (auto-wrapped as "user" role)
auto result = llm.chat("Explain C++ RAII in one sentence.");

// Multi-turn with explicit history management
std::vector<llamalib::Message> messages = {
    {"system", "You are a helpful assistant."},
    {"user", "What is RAII?"},
};
auto r1 = llm.chat(messages);

messages.push_back({"assistant", r1});
messages.push_back({"user", "Give me an example."});
auto r2 = llm.chat(messages);
```

### ChatSession

`ChatSession` manages conversation history automatically.

```cpp
llamalib::ChatSession session(llm, "You are a C++ expert.");
auto r1 = session.say("What is a smart pointer?");
auto r2 = session.say("How does unique_ptr differ from shared_ptr?");

session.clear();  // Reset history (system prompt is preserved)
```

### Streaming

Pass a callback to receive tokens as they are generated. Return `false` from the callback to stop early. Works with both `generate()` and `chat()`.

```cpp
llm.chat("Write a haiku about the sea.", [](std::string_view token) {
    std::cout << token << std::flush;
    return true;  // return false to stop early
});
```

### Custom parameters

Configure context size, temperature, and concurrency via `Options`.

```cpp
llamalib::Options opts;
opts.n_ctx = 4096;
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
void generate(const std::string &prompt, const StreamCallback &callback)
void generate(const std::string &prompt, int max_tokens, const StreamCallback &callback)
```

Raw text completion — sends the prompt directly without chat template formatting. Thread-safe.

#### `chat`

```cpp
// Single message (auto-wrapped as "user" role)
std::string chat(const std::string &message)
std::string chat(const std::string &message, int max_tokens)
void chat(const std::string &message, const StreamCallback &callback)
void chat(const std::string &message, int max_tokens, const StreamCallback &callback)

// Multi-turn
std::string chat(const std::vector<Message> &messages)
std::string chat(const std::vector<Message> &messages, int max_tokens)
void chat(const std::vector<Message> &messages, const StreamCallback &callback)
void chat(const std::vector<Message> &messages, int max_tokens, const StreamCallback &callback)
```

Applies the model's embedded chat template, then generates. Falls back to raw concatenation if the model has no template.

#### Common parameters

- `max_tokens` — maximum tokens to generate; defaults to `-1` (generate until EOS or context limit)
- `callback` — called with each token as it is generated; return `false` to stop early

Throws `std::runtime_error` if the prompt is too long for the context window or if decoding fails.

### `llamalib::Message`

```cpp
struct Message {
    std::string role;     // "system", "user", or "assistant"
    std::string content;
};
```

### `llamalib::ChatSession`

```cpp
ChatSession(Llama &llm, const std::string &system_prompt = "")
```

Wraps `Llama::chat()` with automatic conversation history management.

| Method | Description |
| ------ | ----------- |
| `say(message)` | Appends user message, calls `chat()`, appends assistant reply, returns reply |
| `say(message, max_tokens)` | Same with max_tokens override |
| `say(message, callback)` | Streaming version (void); history is still updated internally |
| `say(message, max_tokens, callback)` | Same with max_tokens override |
| `history()` | Returns `const std::vector<Message>&` of the conversation |
| `clear()` | Clears history (preserves system prompt if set) |

### Type aliases

```cpp
using StreamCallback = std::function<bool(std::string_view token)>;
using SamplerConfig = std::function<void(llama_sampler *chain)>;
```

## License

MIT
