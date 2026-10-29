#include <catch2/catch_test_macros.hpp>

#include "cpp-llamalib.h"

#include <string>
#include <vector>

// MODEL_PATH is defined via CMake's target_compile_definitions
static const char *model_path = MODEL_PATH;

using llamalib::FinishReason;

namespace {

// A grammar fixes what the model writes, so these tests do not depend on what a
// particular model would have said.
llamalib::GenerateOptions fixed(const std::string &text, int max_tokens = 64) {
  llamalib::GenerateOptions o;
  o.max_tokens = max_tokens;
  o.grammar = "root ::= \"" + text + "\"";
  return o;
}

void greedy(llama_sampler *chain) {
  llama_sampler_chain_add(chain, llama_sampler_init_greedy());
}

llamalib::Llama &shared() {
  static llamalib::Llama llm = [] {
    llamalib::Options params;
    params.n_ctx = 512;
    return llamalib::Llama(model_path, params);
  }();
  return llm;
}

} // namespace

TEST_CASE("complete reports how a generation ended", "[result]") {
  auto &llm = shared();

  SECTION("end: the model finished what the grammar allowed") {
    auto r = llm.complete("Say ok", fixed("ok"));
    REQUIRE(r.text == "ok");
    REQUIRE(r.finish == FinishReason::end);
    REQUIRE(r.completion_tokens >= 1);
    REQUIRE(r.completion_tokens <= 2);
  }

  SECTION("length: max_tokens ran out") {
    auto r = llm.complete("Tell me a story", {3});
    REQUIRE(r.finish == FinishReason::length);
    REQUIRE(r.completion_tokens == 3);
    REQUIRE_FALSE(r.text.empty());
  }

  SECTION("stop: a stop sequence appeared, and is not in the text") {
    auto o = fixed("one two three");
    o.stop = {"two"};
    auto r = llm.complete("Count", o);
    REQUIRE(r.finish == FinishReason::stop);
    REQUIRE(r.text == "one ");
  }

  SECTION("aborted: the callback said stop") {
    int calls = 0;
    auto r = llm.complete("Count", fixed("one two three four"),
                          [&](std::string_view) {
                            ++calls;
                            return false;
                          });
    REQUIRE(r.finish == FinishReason::aborted);
    REQUIRE(r.completion_tokens == 1);
    REQUIRE(calls == 1);  // not called again once it said stop
  }

  SECTION("aborted while stop sequences hold text back") {
    auto o = fixed("one two three four");
    o.stop = {"zzz"};
    int calls = 0;
    auto r = llm.complete("Count", o, [&](std::string_view) {
      ++calls;
      return false;
    });
    REQUIRE(r.finish == FinishReason::aborted);
    REQUIRE(calls == 1);  // the held-back text is not sent after the stop
  }
}

TEST_CASE("complete counts tokens", "[result]") {
  auto &llm = shared();
  llm.clear_cache();

  auto r = llm.complete("Hello there", fixed("ok"));
  REQUIRE(r.prompt_tokens == static_cast<int>(llm.token_count("Hello there")));

  // The cache holding the prompt does not change what is reported.
  auto again = llm.complete("Hello there", fixed("ok"));
  REQUIRE(again.prompt_tokens == r.prompt_tokens);
}

TEST_CASE("a stream and the result agree", "[result]") {
  auto &llm = shared();
  std::string streamed;
  auto r = llm.complete("Count", fixed("one two three"),
                        [&](std::string_view piece) {
                          streamed.append(piece);
                          return true;
                        });
  REQUIRE(r.text == "one two three");
  REQUIRE(streamed == r.text);
  REQUIRE(llm.generate("Count", fixed("one two three")) == r.text);
}

TEST_CASE("temperature and seed are per call", "[result][sampling]") {
  auto &llm = shared();
  auto run = [&](llamalib::GenerateOptions o) {
    llm.clear_cache();
    o.max_tokens = 24;
    return llm.complete("Tell me a story", o).text;
  };

  SECTION("temperature 0 repeats") {
    llamalib::GenerateOptions o;
    o.temperature = 0.0f;
    REQUIRE(run(o) == run(o));
  }

  SECTION("a seed makes a warm sample repeat, and another seed differ") {
    llamalib::GenerateOptions a;
    a.temperature = 2.0f;
    a.seed = 1;
    llamalib::GenerateOptions b = a;
    b.seed = 2;
    REQUIRE(run(a) == run(a));
    REQUIRE(run(a) != run(b));
  }

  SECTION("a seed alone makes the call's own chain, at the model's temperature") {
    llamalib::Options hot;
    hot.n_ctx = 512;
    hot.temperature = 2.0f;
    llamalib::Llama warm(model_path, hot);
    auto draw = [&](uint32_t seed) {
      warm.clear_cache();
      llamalib::GenerateOptions o;
      o.max_tokens = 24;
      o.seed = seed;
      return warm.complete("Tell me a story", o).text;
    };
    REQUIRE(draw(1) == draw(1));
    REQUIRE(draw(1) != draw(2));
  }

  SECTION("a sampler chain of its own wins, and greedy is temperature 0") {
    llamalib::GenerateOptions cold;
    cold.temperature = 0.0f;
    llamalib::GenerateOptions chain;
    chain.temperature = 2.0f;  // ignored: the chain wins
    chain.sampler_config = greedy;
    REQUIRE(run(chain) == run(cold));
  }

  SECTION("the model's own sampler is untouched after a call with its own") {
    llamalib::GenerateOptions warm;
    warm.temperature = 2.0f;
    warm.seed = 7;
    run(warm);
    llamalib::Options params;
    params.n_ctx = 512;
    params.temperature = 0.0f;
    llamalib::Llama cold_llm(model_path, params);
    auto before = cold_llm.complete("Tell me a story", {24}).text;
    cold_llm.complete("Tell me a story", [&] {
      llamalib::GenerateOptions o;
      o.max_tokens = 24;
      o.temperature = 2.0f;
      o.seed = 3;
      return o;
    }());
    cold_llm.clear_cache();
    REQUIRE(cold_llm.complete("Tell me a story", {24}).text == before);
  }
}

TEST_CASE("chat_complete and format_chat", "[result][chat]") {
  auto &llm = shared();
  std::vector<llamalib::Message> messages = {
      {"system", "Answer in one word."}, {"user", "Hello"}};

  llamalib::GenerateOptions o;
  o.max_tokens = 16;
  o.temperature = 0.0f;
  o.reuse_cache = false;

  auto prompt = llm.format_chat(messages);
  REQUIRE(prompt != "Answer in one word.Hello");  // the template was applied

  auto via_prompt = llm.complete(prompt, o);
  auto via_chat = llm.chat_complete(messages, o);
  REQUIRE(via_chat.text == via_prompt.text);
  REQUIRE(via_chat.prompt_tokens == via_prompt.prompt_tokens);
  REQUIRE(llm.chat(messages, o) == via_chat.text);
}

TEST_CASE("errors have kinds", "[result][errors]") {
  SECTION("a model that cannot be loaded") {
    REQUIRE_THROWS_AS(llamalib::Llama("/nonexistent/model.gguf"),
                      llamalib::ModelLoadError);
    REQUIRE_THROWS_AS(llamalib::Llama("/nonexistent/model.gguf"),
                      llamalib::Error);
    REQUIRE_THROWS_AS(llamalib::Llama("/nonexistent/model.gguf"),
                      std::runtime_error);
  }

  SECTION("a prompt that does not fit the context") {
    llamalib::Options params;
    params.n_ctx = 64;
    llamalib::Llama small(model_path, params);

    // llama.cpp rounds the context up, so ask what it gave.
    std::string long_prompt;
    for (int i = 0; i < small.n_ctx() + 50; i++) { long_prompt += "word "; }
    REQUIRE_THROWS_AS(small.complete(long_prompt), llamalib::ContextOverflowError);
    REQUIRE_THROWS_AS(small.complete("Hello", {small.n_ctx() + 1}),
                      llamalib::ContextOverflowError);
    REQUIRE_THROWS_AS(small.complete("Hello", {small.n_ctx() + 1}),
                      std::runtime_error);
  }
}

TEST_CASE("options and handles", "[result][options]") {
  SECTION("cpu_only still generates") {
    llamalib::Options params;
    params.n_ctx = 512;
    params.cpu_only = true;
    llamalib::Llama llm(model_path, params);
    REQUIRE(llm.complete("Say ok", fixed("ok")).text == "ok");
  }

  SECTION("the model and the context size are offered") {
    auto &llm = shared();
    REQUIRE(llm.model() != nullptr);
    REQUIRE(llm.n_ctx() == 512);
  }

  SECTION("n_batch is what was asked for") {
    llamalib::Options params;
    params.n_ctx = 512;
    params.n_batch = 32;
    llamalib::Llama llm(model_path, params);
    REQUIRE(llm.n_batch() == 32);
  }

  // Not covered here: cpu_only leaves the same text on a CPU-only build, and
  // strict_chat_template only matters for a model whose template cannot be
  // applied; neither has anything to show with this model and build.

  SECTION("a strict template does not refuse a model that has one") {
    llamalib::Options params;
    params.n_ctx = 512;
    params.strict_chat_template = true;
    llamalib::Llama llm(model_path, params);
    REQUIRE_NOTHROW(llm.format_chat({{"user", "Hello"}}));
  }
}

// A prompt the cache holds is decoded again from its last token alone, and on
// some prompts that is enough to change a greedy answer ("Hello", then
// "Greetings."). These two do on the machine this was written on; where they do
// not, the test cannot tell the option from nothing.
TEST_CASE("the same call gives the same text without the cache", "[result][cache]") {
  auto &llm = shared();
  std::vector<std::vector<llamalib::Message>> chats = {
      {{"system", "Answer in one word."}, {"user", "Hello"}},
      {{"user", "Write a haiku about rain."}},
  };
  for (const auto &messages : chats) {
    llamalib::GenerateOptions o;
    o.max_tokens = 24;
    o.temperature = 0.0f;
    o.reuse_cache = false;

    llm.clear_cache();
    auto first = llm.chat_complete(messages, o);
    auto second = llm.chat_complete(messages, o);  // the prompt is in the cache
    auto third = llm.chat_complete(messages, o);
    REQUIRE(second.text == first.text);
    REQUIRE(third.text == first.text);
    REQUIRE(second.prompt_tokens == first.prompt_tokens);
  }
}

TEST_CASE("a prompt longer than a batch is decoded in pieces", "[result][batch]") {
  llamalib::Options params;
  params.n_ctx = 512;
  params.n_batch = 32;
  llamalib::Llama llm(model_path, params);

  std::string long_prompt;
  for (int i = 0; i < 100; i++) { long_prompt += "word "; }
  REQUIRE(llm.token_count(long_prompt) > 32 * 2);

  // Used to stop the process in a GGML_ASSERT.
  auto r = llm.complete(long_prompt, fixed("ok"));
  REQUIRE(r.text == "ok");
  REQUIRE(r.prompt_tokens == static_cast<int>(llm.token_count(long_prompt)));
}
