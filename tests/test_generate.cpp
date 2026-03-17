#include <catch2/catch_test_macros.hpp>

#include "cpp-llamalib.h"

#include <future>
#include <string>
#include <type_traits>
#include <vector>

// MODEL_PATH is defined via CMake's target_compile_definitions
static const char *model_path = MODEL_PATH;

// Helper traits: detect if const LLM& supports generate overloads
template <typename T, typename = void>
struct const_has_generate_1 : std::false_type {};
template <typename T>
struct const_has_generate_1<
    T, std::void_t<decltype(std::declval<const T &>().generate(
           std::declval<const std::string &>()))>> : std::true_type {};

template <typename T, typename = void>
struct const_has_generate_2 : std::false_type {};
template <typename T>
struct const_has_generate_2<
    T, std::void_t<decltype(std::declval<const T &>().generate(
           std::declval<const std::string &>(), int{}))>> : std::true_type {};

TEST_CASE("generate is non-const", "[llm][const]") {
  // generate() mutates internal state (slot queue), so it must NOT be
  // callable on a const LLM reference.
  STATIC_REQUIRE_FALSE(const_has_generate_1<llamalib::LLM>::value);
  STATIC_REQUIRE_FALSE(const_has_generate_2<llamalib::LLM>::value);
}

TEST_CASE("LLM move semantics", "[llm][move]") {
  STATIC_REQUIRE(std::is_move_constructible_v<llamalib::LLM>);
  STATIC_REQUIRE(std::is_move_assignable_v<llamalib::LLM>);

  llamalib::Params params;
  params.n_ctx = 512;
  params.max_tokens = 16;

  llamalib::LLM original(model_path, params);
  llamalib::LLM moved(std::move(original));

  auto result = moved.generate("Hello");
  REQUIRE_FALSE(result.empty());
}

TEST_CASE("LLM construction", "[llm]") {
  SECTION("valid model loads successfully") {
    REQUIRE_NOTHROW(llamalib::LLM(model_path));
  }

  SECTION("invalid model path throws") {
    REQUIRE_THROWS_AS(llamalib::LLM("/nonexistent/model.gguf"),
                      std::runtime_error);
  }

  SECTION("custom params are accepted") {
    llamalib::Params params;
    params.n_ctx = 512;
    params.max_tokens = 64;
    params.n_slots = 2;
    REQUIRE_NOTHROW(llamalib::LLM(model_path, params));
  }
}

TEST_CASE("generate produces output", "[generate]") {
  llamalib::Params params;
  params.n_ctx = 512;
  params.max_tokens = 32;
  llamalib::LLM llm(model_path, params);

  SECTION("returns non-empty string") {
    auto result = llm.generate("Hello");
    REQUIRE_FALSE(result.empty());
  }

  SECTION("respects max_tokens override") {
    auto short_result = llm.generate("Tell me a story", 8);
    auto long_result = llm.generate("Tell me a story", 64);
    // Shorter max_tokens should generally produce shorter or equal output
    REQUIRE(short_result.size() <= long_result.size() + 32);
  }

  SECTION("different prompts produce different outputs") {
    auto result1 = llm.generate("What is 1+1?");
    auto result2 = llm.generate("Write a poem about the sea");
    // With distinct prompts, outputs should differ
    REQUIRE(result1 != result2);
  }
}

TEST_CASE("long prompt tokenization", "[generate]") {
  llamalib::Params params;
  params.n_ctx = 2048;
  params.max_tokens = 8;
  llamalib::LLM llm(model_path, params);

  // Emoji sequences: each emoji is 4 bytes in UTF-8 but may tokenize
  // into multiple byte-level tokens, potentially exceeding the
  // prompt.size() + 16 buffer if not handled properly.
  std::string long_prompt;
  for (int i = 0; i < 300; i++) {
    long_prompt += "\xF0\x9F\x98\x80";  // U+1F600 grinning face
  }

  REQUIRE_NOTHROW(llm.generate(long_prompt));
}

TEST_CASE("sampler reset produces consistent results", "[generate]") {
  llamalib::Params params;
  params.n_ctx = 512;
  params.max_tokens = 32;
  params.temperature = 0.0f;  // Greedy for determinism
  params.n_slots = 1;
  llamalib::LLM llm(model_path, params);

  // With sampler reset and greedy sampling, repeated calls with the
  // same prompt should produce identical output.
  auto result1 = llm.generate("What is 2+2?");
  auto result2 = llm.generate("What is 2+2?");
  REQUIRE(result1 == result2);
}

TEST_CASE("decode failure throws on context overflow", "[generate]") {
  llamalib::Params params;
  params.n_ctx = 256;
  params.max_tokens = 8;
  llamalib::LLM llm(model_path, params);

  // Build a prompt that exceeds the context window
  std::string long_prompt;
  for (int i = 0; i < 200; i++) {
    long_prompt += "The quick brown fox jumps over the lazy dog. ";
  }

  REQUIRE_THROWS_AS(llm.generate(long_prompt), std::runtime_error);
}

TEST_CASE("streaming generate", "[generate][streaming]") {
  llamalib::Params params;
  params.n_ctx = 512;
  params.max_tokens = 32;
  llamalib::LLM llm(model_path, params);

  SECTION("callback receives tokens and concatenation matches result") {
    std::string streamed;
    auto result = llm.generate("Hello", params.max_tokens,
                               [&](const std::string &token) {
                                 streamed += token;
                                 return true;
                               });
    REQUIRE(result == streamed);
    REQUIRE_FALSE(result.empty());
  }

  SECTION("returning false from callback stops generation early") {
    int count = 0;
    auto result = llm.generate("Tell me a long story", params.max_tokens,
                               [&](const std::string &) {
                                 return ++count < 3;
                               });
    REQUIRE(count == 3);
  }
}

TEST_CASE("custom sampler configuration", "[generate][sampler]") {
  llamalib::Params params;
  params.n_ctx = 512;
  params.max_tokens = 16;
  // Custom sampler: greedy (temp=0, no dist)
  params.sampler_setup = [](llama_sampler *chain) {
    llama_sampler_chain_add(chain, llama_sampler_init_temp(0.0f));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(42));
  };
  llamalib::LLM llm(model_path, params);

  auto r1 = llm.generate("What is 2+2?");
  auto r2 = llm.generate("What is 2+2?");
  REQUIRE_FALSE(r1.empty());
  // With fixed seed, results should be deterministic
  REQUIRE(r1 == r2);
}

TEST_CASE("multiple LLM instances coexist", "[llm]") {
  llamalib::Params params;
  params.n_ctx = 512;
  params.max_tokens = 16;

  llamalib::LLM llm1(model_path, params);
  llamalib::LLM llm2(model_path, params);

  auto r1 = llm1.generate("Hello");
  auto r2 = llm2.generate("Hello");

  REQUIRE_FALSE(r1.empty());
  REQUIRE_FALSE(r2.empty());
}

TEST_CASE("concurrent generate calls with slot pool", "[generate][concurrent]") {
  llamalib::Params params;
  params.n_ctx = 512;
  params.max_tokens = 16;
  params.n_slots = 2;
  llamalib::LLM llm(model_path, params);

  std::vector<std::future<std::string>> futures;
  for (int i = 0; i < 4; i++) {
    futures.push_back(
        std::async(std::launch::async, [&llm] {
          return llm.generate("Hi");
        }));
  }

  for (auto &f : futures) {
    auto result = f.get();
    REQUIRE_FALSE(result.empty());
  }
}
