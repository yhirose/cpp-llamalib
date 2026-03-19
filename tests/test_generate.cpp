#include <catch2/catch_test_macros.hpp>

#include "cpp-llamalib.h"

#include <future>
#include <string>
#include <type_traits>
#include <vector>

// MODEL_PATH is defined via CMake's target_compile_definitions
static const char *model_path = MODEL_PATH;

TEST_CASE("Llama move semantics", "[llm][move]") {
  STATIC_REQUIRE(std::is_move_constructible_v<llamalib::Llama>);
  STATIC_REQUIRE(std::is_move_assignable_v<llamalib::Llama>);

  llamalib::Options params;
  params.n_ctx = 512;

  llamalib::Llama original(model_path, params);
  llamalib::Llama moved(std::move(original));

  auto result = moved.generate("Hello", {16});
  REQUIRE_FALSE(result.empty());
}

TEST_CASE("Llama construction", "[llm]") {
  SECTION("valid model loads successfully") {
    REQUIRE_NOTHROW(llamalib::Llama(model_path));
  }

  SECTION("invalid model path throws") {
    REQUIRE_THROWS_AS(llamalib::Llama("/nonexistent/model.gguf"),
                      std::runtime_error);
  }

  SECTION("custom params are accepted") {
    llamalib::Options params;
    params.n_ctx = 512;
    params.n_slots = 2;
    REQUIRE_NOTHROW(llamalib::Llama(model_path, params));
  }
}

TEST_CASE("generate produces output", "[generate]") {
  llamalib::Options params;
  params.n_ctx = 512;
  llamalib::Llama llm(model_path, params);

  SECTION("returns non-empty string") {
    auto result = llm.generate("Hello", {32});
    REQUIRE_FALSE(result.empty());
  }

  SECTION("respects max_tokens") {
    auto short_result = llm.generate("Tell me a story", {8});
    auto long_result = llm.generate("Tell me a story", {64});
    // Shorter max_tokens should generally produce shorter or equal output
    REQUIRE(short_result.size() <= long_result.size() + 32);
  }

  SECTION("different prompts produce different outputs") {
    auto result1 = llm.generate("What is 1+1?", {32});
    auto result2 = llm.generate("Write a poem about the sea", {32});
    REQUIRE(result1 != result2);
  }
}

TEST_CASE("long prompt tokenization", "[generate]") {
  llamalib::Options params;
  params.n_ctx = 2048;
  llamalib::Llama llm(model_path, params);

  // Emoji sequences: each emoji is 4 bytes in UTF-8 but may tokenize
  // into multiple byte-level tokens, potentially exceeding the
  // prompt.size() + 16 buffer if not handled properly.
  std::string long_prompt;
  for (int i = 0; i < 300; i++) {
    long_prompt += "\xF0\x9F\x98\x80";  // U+1F600 grinning face
  }

  REQUIRE_NOTHROW(llm.generate(long_prompt, {8}));
}

TEST_CASE("sampler reset produces consistent results", "[generate]") {
  llamalib::Options params;
  params.n_ctx = 512;
  params.temperature = 0.0f;  // Greedy for determinism
  params.n_slots = 1;
  llamalib::Llama llm(model_path, params);

  auto result1 = llm.generate("What is 2+2?", {32});
  auto result2 = llm.generate("What is 2+2?", {32});
  REQUIRE(result1 == result2);
}

TEST_CASE("decode failure throws on context overflow", "[generate]") {
  llamalib::Options params;
  params.n_ctx = 256;
  llamalib::Llama llm(model_path, params);

  std::string long_prompt;
  for (int i = 0; i < 200; i++) {
    long_prompt += "The quick brown fox jumps over the lazy dog. ";
  }

  REQUIRE_THROWS_AS(llm.generate(long_prompt), std::runtime_error);
}

TEST_CASE("streaming generate", "[generate][streaming]") {
  llamalib::Options params;
  params.n_ctx = 512;
  llamalib::Llama llm(model_path, params);

  SECTION("callback receives tokens") {
    std::string streamed;
    llm.generate("Hello", {32}, [&](std::string_view token) {
      streamed += token;
      return true;
    });
    REQUIRE_FALSE(streamed.empty());
  }

  SECTION("returning false from callback stops generation early") {
    int count = 0;
    llm.generate("Tell me a long story", {32}, [&](std::string_view) {
      return ++count < 3;
    });
    REQUIRE(count == 3);
  }
}

TEST_CASE("custom sampler configuration", "[generate][sampler]") {
  llamalib::Options params;
  params.n_ctx = 512;
  // Custom sampler: greedy (temp=0) with fixed seed
  params.sampler_config = [](llama_sampler *chain) {
    llama_sampler_chain_add(chain, llama_sampler_init_temp(0.0f));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(42));
  };
  llamalib::Llama llm(model_path, params);

  auto r1 = llm.generate("What is 2+2?", {16});
  auto r2 = llm.generate("What is 2+2?", {16});
  REQUIRE_FALSE(r1.empty());
  // With fixed seed, results should be deterministic
  REQUIRE(r1 == r2);
}

TEST_CASE("multiple Llama instances coexist", "[llm]") {
  llamalib::Options params;
  params.n_ctx = 512;

  llamalib::Llama llm1(model_path, params);
  llamalib::Llama llm2(model_path, params);

  auto r1 = llm1.generate("Hello", {16});
  auto r2 = llm2.generate("Hello", {16});

  REQUIRE_FALSE(r1.empty());
  REQUIRE_FALSE(r2.empty());
}

TEST_CASE("model has an embedded chat template", "[chat_template]") {
  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  llama_model_ptr model{llama_model_load_from_file(model_path, model_params)};
  REQUIRE(model != nullptr);
  auto tmpl = llama_model_chat_template(model.get(), nullptr);
  REQUIRE(tmpl != nullptr);
  REQUIRE(std::string_view(tmpl).size() > 0);
}

TEST_CASE("chat applies template", "[chat][chat_template]") {
  llamalib::Options params;
  params.n_ctx = 512;
  params.temperature = 0.0f;
  llamalib::Llama llm(model_path, params);

  SECTION("single message chat produces output") {
    auto result = llm.chat("What is 2+2?", {32});
    REQUIRE_FALSE(result.empty());
  }

  SECTION("deterministic chat output") {
    auto r1 = llm.chat("Say hello", {32});
    auto r2 = llm.chat("Say hello", {32});
    REQUIRE(r1 == r2);
  }

  SECTION("template-wrapped prompt expands token count") {
    llamalib::Options tight;
    tight.n_ctx = 256;
    llamalib::Llama tight_llm(model_path, tight);

    std::string prompt;
    for (int i = 0; i < 60; i++) {
      prompt += "The quick brown fox. ";
    }

    REQUIRE_THROWS_AS(tight_llm.chat(prompt), std::runtime_error);
  }
}

TEST_CASE("chat with multi-turn messages", "[chat][chat_template]") {
  llamalib::Options params;
  params.n_ctx = 512;
  params.temperature = 0.0f;
  llamalib::Llama llm(model_path, params);

  SECTION("multi-turn conversation produces output") {
    std::vector<llamalib::Message> messages = {
        {"user", "My name is Alice."},
        {"assistant", "Hello Alice! How can I help you?"},
        {"user", "What is my name?"},
    };
    auto result = llm.chat(messages, {32});
    REQUIRE_FALSE(result.empty());
  }

  SECTION("system message is supported") {
    std::vector<llamalib::Message> messages = {
        {"system", "You are a helpful assistant."},
        {"user", "Hello"},
    };
    auto result = llm.chat(messages, {32});
    REQUIRE_FALSE(result.empty());
  }
}

TEST_CASE("ChatSession wrapper", "[conversation]") {
  llamalib::Options params;
  params.n_ctx = 512;
  llamalib::Llama llm(model_path, params);

  SECTION("basic multi-turn conversation") {
    auto conv = llm.session("You are a helpful assistant.");
    auto r1 = conv.say("Hello", {32});
    REQUIRE_FALSE(r1.empty());
    auto r2 = conv.say("How are you?", {32});
    REQUIRE_FALSE(r2.empty());
  }

  SECTION("history is accumulated correctly") {
    auto conv = llm.session("System prompt.");
    conv.say("First message", {16});
    conv.say("Second message", {16});
    auto &h = conv.history();
    REQUIRE(h.size() == 5); // system + 2*(user + assistant)
    REQUIRE(h[0].role == "system");
    REQUIRE(h[1].role == "user");
    REQUIRE(h[2].role == "assistant");
    REQUIRE(h[3].role == "user");
    REQUIRE(h[4].role == "assistant");
  }

  SECTION("clear preserves system prompt") {
    auto conv = llm.session("Keep this.");
    conv.say("Hello", {16});
    REQUIRE(conv.history().size() == 3);
    conv.clear();
    REQUIRE(conv.history().size() == 1);
    REQUIRE(conv.history()[0].role == "system");
    REQUIRE(conv.history()[0].content == "Keep this.");
  }

  SECTION("clear without system prompt") {
    auto conv = llm.session();
    conv.say("Hello", {16});
    REQUIRE(conv.history().size() == 2);
    conv.clear();
    REQUIRE(conv.history().empty());
  }

}

TEST_CASE("KV cache reuse produces correct results", "[generate][kv_cache]") {
  llamalib::Options params;
  params.n_ctx = 512;
  params.temperature = 0.0f;
  params.n_slots = 1;
  llamalib::Llama llm(model_path, params);

  SECTION("same prompt twice gives identical results with cache reuse") {
    auto r1 = llm.generate("What is 2+2?", {32});
    auto r2 = llm.generate("What is 2+2?", {32});
    REQUIRE_FALSE(r1.empty());
    REQUIRE(r1 == r2);
  }

  SECTION("different prompt after cached prompt works correctly") {
    auto r1 = llm.generate("Hello world", {16});
    auto r2 = llm.generate("Goodbye world", {16});
    REQUIRE_FALSE(r1.empty());
    REQUIRE_FALSE(r2.empty());
  }

  SECTION("shared prefix reuses cache") {
    auto r1 = llm.generate("The quick brown fox jumps", {16});
    auto r2 = llm.generate("The quick brown fox runs", {16});
    REQUIRE_FALSE(r1.empty());
    REQUIRE_FALSE(r2.empty());
  }

  SECTION("clear_cache resets state") {
    llm.generate("Hello", {16});
    llm.clear_cache();
    auto r = llm.generate("Hello", {16});
    REQUIRE_FALSE(r.empty());
  }
}

TEST_CASE("ChatSession with KV cache reuse", "[conversation][kv_cache]") {
  llamalib::Options params;
  params.n_ctx = 1024;
  params.temperature = 0.0f;
  llamalib::Llama llm(model_path, params);

  llamalib::ChatSession conv(llm, "You are a helpful assistant.");
  auto r1 = conv.say("My name is Bob.", {32});
  REQUIRE_FALSE(r1.empty());
  auto r2 = conv.say("What did I just tell you?", {32});
  REQUIRE_FALSE(r2.empty());
  auto r3 = conv.say("Say goodbye.", {32});
  REQUIRE_FALSE(r3.empty());
}

TEST_CASE("structured output with GBNF grammar", "[generate][grammar]") {
  llamalib::Options params;
  params.n_ctx = 512;
  params.temperature = 0.0f;
  llamalib::Llama llm(model_path, params);

  // Simple grammar: only yes or no
  std::string yesno_grammar = "root ::= \"yes\" | \"no\"";

  SECTION("grammar constrains output") {
    llamalib::GenerateOptions opts;
    opts.max_tokens = 8;
    opts.grammar = yesno_grammar;

    auto result = llm.chat("Is the sky blue? Answer only yes or no.", opts);
    REQUIRE_FALSE(result.empty());
    REQUIRE((result == "yes" || result == "no"));
  }

  SECTION("generate without grammar still works") {
    llamalib::GenerateOptions opts;
    opts.max_tokens = 16;
    auto result = llm.generate("Hello", opts);
    REQUIRE_FALSE(result.empty());
  }

  SECTION("GenerateOptions with callback") {
    llamalib::GenerateOptions opts;
    opts.max_tokens = 16;
    std::string streamed;
    llm.generate("Hello", opts, [&](std::string_view token) {
      streamed += token;
      return true;
    });
    REQUIRE_FALSE(streamed.empty());
  }
}

TEST_CASE("ChatSession with GenerateOptions", "[conversation][grammar]") {
  llamalib::Options params;
  params.n_ctx = 512;
  params.temperature = 0.0f;
  llamalib::Llama llm(model_path, params);

  auto conv = llm.session();
  llamalib::GenerateOptions opts;
  opts.max_tokens = 32;
  auto result = conv.say("Hello", opts);
  REQUIRE_FALSE(result.empty());
  REQUIRE(conv.history().size() == 2);
}

TEST_CASE("concurrent generate calls with slot pool", "[generate][concurrent]") {
  llamalib::Options params;
  params.n_ctx = 512;
  params.n_slots = 2;
  llamalib::Llama llm(model_path, params);

  std::vector<std::future<std::string>> futures;
  for (int i = 0; i < 4; i++) {
    futures.push_back(
        std::async(std::launch::async, [&llm] {
          return llm.generate("Hi", {16});
        }));
  }

  for (auto &f : futures) {
    auto result = f.get();
    REQUIRE_FALSE(result.empty());
  }
}
