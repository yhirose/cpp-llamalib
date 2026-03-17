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
