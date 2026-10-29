# cpp-llamalib

[![CI](https://github.com/yhirose/cpp-llamalib/actions/workflows/ci.yml/badge.svg)](https://github.com/yhirose/cpp-llamalib/actions/workflows/ci.yml)

A C++17 single-file header-only wrapper for [llama.cpp](https://github.com/ggml-org/llama.cpp).<br>
Just include **cpp-llamalib.h** to call llama.cpp with a simple, high-level API.

This library is used in ["Building a Desktop LLM App with cpp-httplib"](https://yhirose.github.io/cpp-httplib/en/llm-app/).

## Features

- Header-only — single file, no build step for the wrapper itself
- Two-tier API: raw `generate()` and template-aware `chat()`
- `ChatSession` wrapper for multi-turn conversations
- KV cache reuse — shared prompt prefixes are not re-processed
- Stop sequences — halt generation when a specified string appears
- Tokenize API — count tokens before generation to check context limits
- Structured output via GBNF grammar constraints
- Thread-safe concurrent generation via slot pool
- Custom sampler configuration
- `Result` with the finish reason and token counts; per-call temperature, seed and sampler
- Typed errors (`ModelLoadError`, `ContextOverflowError`, ...), all `std::runtime_error`

## Integration

Please copy `cpp-llamalib.h` into your project and `#include` it.

Your project must link against llama.cpp (`llama` and `ggml` libraries).<br>
Tested against llama.cpp [v0.7.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.7.0), 2026-10-15.

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

`ChatSession` manages conversation history automatically. Create one via `Llama::session()`.

```cpp
auto session = llm.session("You are a C++ expert.");
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

### Structured output (GBNF grammar)

Use `GenerateOptions` with a GBNF grammar string to constrain model output. Works with `generate()`, `chat()`, and `ChatSession::say()`.

```cpp
llamalib::GenerateOptions opts;
opts.max_tokens = 32;
opts.grammar = R"(root ::= "yes" | "no")";

auto answer = llm.chat("Is the sky blue?", opts);
// answer is guaranteed to be "yes" or "no"
```

### Stop sequences

Use `GenerateOptions::stop` to halt generation when any of the specified strings appears. The stop sequence itself is excluded from the output. Works with `generate()`, `chat()`, and streaming.

```cpp
llamalib::GenerateOptions opts;
opts.max_tokens = 128;
opts.stop = {"\n", "。"};

auto result = llm.chat("List one item:", opts);
// Generation stops at the first newline or period
```

### Tokenize

Use `tokenize()` or `token_count()` to inspect token counts before generation — useful for checking context limits.

```cpp
auto tokens = llm.tokenize("Hello, world!");  // std::vector<llama_token>
auto count = llm.token_count("Hello, world!"); // count == tokens.size()

// Check context limits before generating
if (llm.token_count(prompt) > 1800) {
    // Truncate or split the prompt
}
```

### KV cache reuse

KV cache is automatically reused across calls on the same slot. When consecutive prompts share a common prefix (e.g. multi-turn chat), only the new tokens are decoded — previous KV entries are kept. Use `clear_cache()` to explicitly reset all slots.

A prompt the cache already holds is decoded again from its last token alone, which can differ in the last digits from decoding it whole, and on some prompts that changes even a greedy answer (`Hello`, then `Greetings.`). When the same prompt and settings must give the same text however many calls came before, set `GenerateOptions::reuse_cache = false` — the call then starts from an empty cache.

```cpp
llm.generate("The quick brown fox jumps", {16});  // full decode
llm.generate("The quick brown fox runs", {16});   // only re-decodes "runs"

llm.clear_cache();  // force full re-processing on next call
```

### Result: how a generation ended

`complete()` and `chat_complete()` return a `Result` — the text, why it ended, and the token counts. With a callback, the text is streamed to it as well.

```cpp
auto r = llm.complete("Hello", {32});
r.text;                 // the generated text
r.finish;               // FinishReason::end, length, stop or aborted
r.prompt_tokens;        // the prompt's tokens, whether or not the cache held them
r.completion_tokens;    // tokens generated, not counting the end-of-generation token
```

| `FinishReason` | Meaning |
| -------------- | ------- |
| `end` | the model produced an end-of-generation token |
| `length` | `max_tokens` (or the context) ran out |
| `stop` | a stop sequence appeared; it is not in the text |
| `aborted` | the stream callback returned `false` (it is not called again) |

### Per-call sampling

`temperature` and `seed` in `GenerateOptions` apply to one call. With either, the call samples with a chain of its own (temperature, then a seeded draw) instead of the one the model was made with; the temperature defaults to `Options::temperature`. A `sampler_config` in `GenerateOptions` replaces the whole chain for the call and wins over both.

```cpp
llamalib::GenerateOptions o;
o.temperature = 0.0f;                 // greedy, for this call
auto a = llm.generate("Hello", o);

o.temperature = 1.0f;
o.seed = 42;                          // the same seed gives the same draw
auto b = llm.generate("Hello", o);
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

Everything this library throws is a `llamalib::Error`, which is a `std::runtime_error`, so catching `std::runtime_error` still works.

| Type | When |
| ---- | ---- |
| `ModelLoadError` | the model file cannot be loaded, or a context cannot be made for it |
| `ContextOverflowError` | the prompt, with the tokens asked for, does not fit the context |
| `TemplateError` | the chat template cannot be applied (only with `Options::strict_chat_template`) |
| `DecodeError` | `llama_decode` failed |
| `Error` | anything else (e.g. tokenization failed) |

```cpp
try {
    llamalib::Llama llm("/bad/path.gguf");
} catch (const llamalib::ModelLoadError &e) {
    // "Failed to load model"
}

try {
    // Throws if prompt + max_tokens exceeds context size
    auto result = llm.generate(very_long_prompt);
} catch (const llamalib::ContextOverflowError &e) {
    // "Prompt too long: 1024 prompt tokens + 512 max_tokens exceeds context size 1024"
}
```

llama.cpp rounds the context size up (to a multiple of 256), so ask `llm.n_ctx()` for the size the model got. A prompt longer than `n_batch` is decoded in pieces of that size.

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
| `cpu_only` | `bool` | `false` | Leave the GPUs out of the devices the model may use |
| `strict_chat_template` | `bool` | `false` | Throw `TemplateError` when the chat template cannot be applied, instead of joining the messages' contents |
| `n_batch` | `int` | `0` | Tokens decoded at a time (0 = llama.cpp's default); a longer prompt is decoded in pieces |

### `llamalib::GenerateOptions`

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `max_tokens` | `int` | `-1` | Maximum tokens to generate (-1 = until EOS or context limit) |
| `grammar` | `std::string` | `""` | GBNF grammar string (empty = no constraint) |
| `grammar_root` | `std::string` | `""` | Root rule name (defaults to `"root"`) |
| `stop` | `std::vector<std::string>` | `{}` | Stop sequences (generation stops when any appears; empty strings are ignored) |
| `temperature` | `std::optional<float>` | none | Temperature for this call (see Per-call sampling) |
| `seed` | `std::optional<uint32_t>` | none | Seed for this call's draw |
| `sampler_config` | `SamplerConfig` | `nullptr` | The whole sampler chain for this call |
| `reuse_cache` | `bool` | `true` | `false` starts the call from an empty cache |

### `llamalib::Llama`

#### Constructor

```cpp
Llama(const std::string &model_path, const Options &params = {})
```

Loads the model and creates context/sampler slots. Throws `ModelLoadError` (a `std::runtime_error`) on failure.

#### `complete` and `chat_complete`

```cpp
Result complete(const std::string &prompt, const GenerateOptions &opts = {}, const StreamCallback &callback = nullptr)
Result chat_complete(const std::vector<Message> &messages, const GenerateOptions &opts = {}, const StreamCallback &callback = nullptr)
```

Like `generate` and `chat`, and they return a `Result`. `generate` and `chat` are these with the text taken out.

#### `format_chat`

```cpp
std::string format_chat(const std::vector<Message> &messages) const
```

The prompt the model's chat template makes of the messages, ready for `generate`, `complete` or `token_count`.

#### `model`, `n_ctx`, `n_batch`

```cpp
const llama_model *model() const
int n_ctx() const    // the context size the model got, which may be rounded up from Options::n_ctx
int n_batch() const  // tokens decoded at a time
```

The model is offered for what this library does not wrap. The contexts are not: each belongs to a slot that a running call holds.

#### `generate`

```cpp
std::string generate(const std::string &prompt, const GenerateOptions &opts = {})
void generate(const std::string &prompt, const StreamCallback &callback)
void generate(const std::string &prompt, const GenerateOptions &opts, const StreamCallback &callback)
```

Raw text completion — sends the prompt directly without chat template formatting. Thread-safe. Throws `ContextOverflowError` if the prompt is too long for the context window, `DecodeError` if decoding fails.

#### `chat`

```cpp
// Single message (auto-wrapped as "user" role)
std::string chat(const std::string &message, const GenerateOptions &opts = {})
void chat(const std::string &message, const StreamCallback &callback)
void chat(const std::string &message, const GenerateOptions &opts, const StreamCallback &callback)

// Multi-turn
std::string chat(const std::vector<Message> &messages, const GenerateOptions &opts = {})
void chat(const std::vector<Message> &messages, const StreamCallback &callback)
void chat(const std::vector<Message> &messages, const GenerateOptions &opts, const StreamCallback &callback)
```

Applies the model's embedded chat template, then generates. Falls back to raw concatenation if the model has no template (or throws `TemplateError` with `Options::strict_chat_template`). Throws `ContextOverflowError` if the prompt is too long for the context window, `DecodeError` if decoding fails.

#### `tokenize`

```cpp
std::vector<llama_token> tokenize(const std::string &text) const
```

Tokenizes the given text and returns the token IDs.

#### `token_count`

```cpp
size_t token_count(const std::string &text) const
```

Returns the number of tokens in the given text. Equivalent to `tokenize(text).size()`.

#### `clear_cache`

```cpp
void clear_cache()
```

Clears the KV cache and cached token history for all slots. Use when you want to force full prompt re-processing on the next call.

### `llamalib::Message`

```cpp
struct Message {
    std::string role;     // "system", "user", or "assistant"
    std::string content;
};
```

### `llamalib::ChatSession`

Created via `Llama::session()`:

```cpp
auto session = llm.session("You are a helpful assistant.");
```

Wraps `Llama::chat()` with automatic conversation history management.

| Method | Description |
| ------ | ----------- |
| `say(message, opts = {})` | Appends user message, calls `chat()`, appends assistant reply, returns reply |
| `say(message, callback)` | Streaming version (void); history is still updated internally |
| `say(message, opts, callback)` | Streaming with options |
| `history()` | Returns `const std::vector<Message>&` of the conversation |
| `clear()` | Clears history (preserves system prompt if set) |

### Type aliases

```cpp
using StreamCallback = std::function<bool(std::string_view token)>;
using SamplerConfig = std::function<void(llama_sampler *chain)>;
```

## License

MIT
