#ifndef CPP_LLAMALIB_H
#define CPP_LLAMALIB_H

#include "llama.h"
#include "llama-cpp.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace llamalib {

namespace detail {
inline std::atomic<int> &backend_refcount() {
  static std::atomic<int> count{0};
  return count;
}
inline void backend_init() {
  if (backend_refcount().fetch_add(1) == 0) { llama_backend_init(); }
}
inline void backend_free() {
  if (backend_refcount().fetch_sub(1) == 1) { llama_backend_free(); }
}
} // namespace detail

// Callback to stream tokens during generation. Return false to stop early.
using StreamCallback = std::function<bool(std::string_view token)>;

// Callback to configure the sampler chain. Receives an empty chain;
// add samplers (e.g. temp, dist, penalties) as needed.
using SamplerConfig = std::function<void(llama_sampler *chain)>;

struct Message {
  std::string role;
  std::string content;
};

struct Options {
  int n_gpu_layers = 99;
  int n_ctx = 2048;
  float temperature = 0.3f;
  int n_slots = 1;
  SamplerConfig sampler_config = nullptr;
};

class Llama {
public:
  Llama(const std::string &model_path, const Options &params = {})
      : params_(params) {
    detail::backend_init();

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;
    model_ = llama_model_ptr{
        llama_model_load_from_file(model_path.c_str(), model_params)};
    if (!model_) { throw std::runtime_error("Failed to load model"); }

    for (auto i = 0; i < params.n_slots; i++) {
      auto ctx_params = llama_context_default_params();
      ctx_params.n_ctx = params.n_ctx;
      auto ctx =
          llama_context_ptr{llama_init_from_model(model_.get(), ctx_params)};
      if (!ctx) { throw std::runtime_error("Failed to create context"); }

      auto smpl = llama_sampler_ptr{
          llama_sampler_chain_init(llama_sampler_chain_default_params())};
      if (params.sampler_config) {
        params.sampler_config(smpl.get());
      } else {
        llama_sampler_chain_add(smpl.get(),
                                llama_sampler_init_temp(params.temperature));
        llama_sampler_chain_add(smpl.get(),
                                llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
      }

      slots_.push({std::move(ctx), std::move(smpl)});
    }
  }

  ~Llama() {
    if (model_) { detail::backend_free(); }
  }

  Llama(const Llama &) = delete;
  Llama &operator=(const Llama &) = delete;

  Llama(Llama &&other) noexcept
      : params_(other.params_),
        model_(std::move(other.model_)),
        slots_(std::move(other.slots_)) {}

  Llama &operator=(Llama &&other) noexcept {
    if (this != &other) {
      // Destroy slots before model (contexts reference the model).
      slots_ = {};
      if (model_) { detail::backend_free(); }
      model_.reset();
      params_ = other.params_;
      model_ = std::move(other.model_);
      slots_ = std::move(other.slots_);
    }
    return *this;
  }

  std::string generate(const std::string &prompt) {
    return generate(prompt, -1);
  }

  std::string generate(const std::string &prompt, int max_tokens) {
    auto guard = slot_guard();
    return run_inference(guard.slot, prompt, max_tokens, nullptr);
  }

  void generate(const std::string &prompt, const StreamCallback &callback) {
    generate(prompt, -1, callback);
  }

  void generate(const std::string &prompt, int max_tokens,
                const StreamCallback &callback) {
    auto guard = slot_guard();
    run_inference(guard.slot, prompt, max_tokens, callback);
  }

  // Tier 2: Chat API (applies chat template)

  std::string chat(const std::string &message) {
    return chat(std::vector<Message>{{"user", message}});
  }

  std::string chat(const std::string &message, int max_tokens) {
    return chat(std::vector<Message>{{"user", message}}, max_tokens);
  }

  void chat(const std::string &message, const StreamCallback &callback) {
    chat(std::vector<Message>{{"user", message}}, callback);
  }

  void chat(const std::string &message, int max_tokens,
            const StreamCallback &callback) {
    chat(std::vector<Message>{{"user", message}}, max_tokens, callback);
  }

  std::string chat(const std::vector<Message> &messages) {
    return chat(messages, -1);
  }

  std::string chat(const std::vector<Message> &messages, int max_tokens) {
    auto formatted = apply_chat_template(messages);
    return generate(formatted, max_tokens);
  }

  void chat(const std::vector<Message> &messages,
            const StreamCallback &callback) {
    chat(messages, -1, callback);
  }

  void chat(const std::vector<Message> &messages, int max_tokens,
            const StreamCallback &callback) {
    auto formatted = apply_chat_template(messages);
    generate(formatted, max_tokens, callback);
  }

private:
  struct Slot {
    llama_context_ptr ctx;
    llama_sampler_ptr smpl;
  };

  struct SlotGuard {
    Slot slot;
    Llama &owner;
    explicit SlotGuard(Llama &o) : slot(o.acquire_slot()), owner(o) {}
    ~SlotGuard() { owner.release_slot(std::move(slot)); }
    SlotGuard(const SlotGuard &) = delete;
    SlotGuard &operator=(const SlotGuard &) = delete;
  };

  SlotGuard slot_guard() { return SlotGuard(*this); }

  Slot acquire_slot() {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [&] { return !slots_.empty(); });
    auto slot = std::move(slots_.front());
    slots_.pop();
    return slot;
  }

  void release_slot(Slot slot) {
    std::lock_guard lock(mutex_);
    slots_.push(std::move(slot));
    cv_.notify_one();
  }

  static std::string concat_contents(const std::vector<Message> &messages) {
    size_t total = 0;
    for (const auto &m : messages) { total += m.content.size(); }
    std::string result;
    result.reserve(total);
    for (const auto &m : messages) { result += m.content; }
    return result;
  }

  std::string apply_chat_template(const std::vector<Message> &messages) const {
    auto tmpl = llama_model_chat_template(model_.get(), nullptr);
    if (!tmpl) { return concat_contents(messages); }

    // Convert Message structs to llama_chat_message (pointers into messages)
    std::vector<llama_chat_message> chat_msgs;
    chat_msgs.reserve(messages.size());
    for (const auto &m : messages) {
      chat_msgs.push_back({m.role.c_str(), m.content.c_str()});
    }

    auto len = llama_chat_apply_template(
        tmpl, chat_msgs.data(), chat_msgs.size(), true, nullptr, 0);
    if (len < 0) { return concat_contents(messages); }

    std::string formatted(static_cast<size_t>(len), '\0');
    auto written = llama_chat_apply_template(
        tmpl, chat_msgs.data(), chat_msgs.size(), true, formatted.data(),
        formatted.size());
    if (written < 0) { return concat_contents(messages); }
    formatted.resize(static_cast<size_t>(written));
    return formatted;
  }

  std::string run_inference(Slot &slot, const std::string &prompt,
                            int max_tokens, const StreamCallback &callback) {
    llama_sampler_reset(slot.smpl.get());
    auto vocab = llama_model_get_vocab(model_.get());

    std::vector<llama_token> tokens(prompt.size() + 16);
    auto tokenize = [&] {
      return llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(),
                            tokens.size(), true, true);
    };
    auto n = tokenize();
    if (n < 0) {
      // Buffer was too small; -n is the required size
      tokens.resize(static_cast<size_t>(-n));
      n = tokenize();
      if (n < 0) {
        throw std::runtime_error("Failed to tokenize prompt");
      }
    }
    tokens.resize(static_cast<size_t>(n));

    auto n_ctx = llama_n_ctx(slot.ctx.get());
    if (max_tokens < 0) {
      // -1 means generate until EOS or context limit
      max_tokens = static_cast<int>(n_ctx - tokens.size());
      if (max_tokens <= 0) {
        throw std::runtime_error(
            "Prompt too long: " + std::to_string(tokens.size()) +
            " prompt tokens fills the entire context of " +
            std::to_string(n_ctx));
      }
    } else if (tokens.size() + max_tokens > n_ctx) {
      throw std::runtime_error(
          "Prompt too long: " + std::to_string(tokens.size()) +
          " prompt tokens + " + std::to_string(max_tokens) +
          " max_tokens exceeds context size " + std::to_string(n_ctx));
    }

    llama_memory_clear(llama_get_memory(slot.ctx.get()), true);
    if (llama_decode(slot.ctx.get(),
                     llama_batch_get_one(tokens.data(), tokens.size()))) {
      throw std::runtime_error("llama_decode failed on prompt");
    }

    std::string result;
    std::string piece;
    for (auto i = 0; i < max_tokens; i++) {
      auto new_token =
          llama_sampler_sample(slot.smpl.get(), slot.ctx.get(), -1);
      if (llama_vocab_is_eog(vocab, new_token)) break;

      char buf[256];
      auto len =
          llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
      if (len < 0) {
        piece.resize(static_cast<size_t>(-len));
        len = llama_token_to_piece(vocab, new_token, piece.data(),
                                   piece.size(), 0, true);
      }
      if (len <= 0) continue;

      std::string_view sv = (len <= static_cast<int>(sizeof(buf)))
                                 ? std::string_view(buf, static_cast<size_t>(len))
                                 : std::string_view(piece.data(), static_cast<size_t>(len));
      if (callback) {
        if (!callback(sv)) break;
      } else {
        result.append(sv);
      }

      if (llama_decode(slot.ctx.get(), llama_batch_get_one(&new_token, 1))) {
        throw std::runtime_error("llama_decode failed during generation");
      }
    }
    return result;
  }

  Options params_;
  llama_model_ptr model_;
  std::queue<Slot> slots_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

class ChatSession {
public:
  ChatSession(Llama &llm, const std::string &system_prompt = "")
      : llm_(llm) {
    if (!system_prompt.empty()) {
      history_.push_back({"system", system_prompt});
    }
  }

  std::string say(const std::string &message) {
    return say_impl(message, [&] { return llm_.chat(history_); });
  }

  std::string say(const std::string &message, int max_tokens) {
    return say_impl(message, [&] { return llm_.chat(history_, max_tokens); });
  }

  void say(const std::string &message, const StreamCallback &callback) {
    say(message, -1, callback);
  }

  void say(const std::string &message, int max_tokens,
           const StreamCallback &callback) {
    say_impl(message, [&] {
      std::string reply;
      llm_.chat(history_, max_tokens, [&](std::string_view token) {
        reply += token;
        return callback(token);
      });
      return reply;
    });
  }

  const std::vector<Message> &history() const { return history_; }

  void clear() {
    std::string system_prompt;
    if (!history_.empty() && history_.front().role == "system") {
      system_prompt = std::move(history_.front().content);
    }
    history_.clear();
    if (!system_prompt.empty()) {
      history_.push_back({"system", system_prompt});
    }
  }

private:
  template <typename ChatFn>
  std::string say_impl(const std::string &message, ChatFn &&chat_fn) {
    history_.push_back({"user", message});
    try {
      auto reply = chat_fn();
      history_.push_back({"assistant", reply});
      return reply;
    } catch (...) {
      history_.pop_back();
      throw;
    }
  }

  Llama &llm_;
  std::vector<Message> history_;
};

} // namespace llamalib

#endif // CPP_LLAMALIB_H
