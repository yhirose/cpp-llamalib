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

struct GenerateOptions {
  int max_tokens = -1;
  std::string grammar;       // GBNF grammar string (empty = no constraint)
  std::string grammar_root;  // Root rule name (default "root")
  std::vector<std::string> stop;  // Stop sequences (generation stops when any appears)
};

struct Options {
  int n_gpu_layers = 99;
  int n_ctx = 2048;
  float temperature = 0.3f;
  int n_slots = 1;
  SamplerConfig sampler_config = nullptr;
};

class ChatSession;

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

  std::string generate(const std::string &prompt,
                       const GenerateOptions &opts = {}) {
    auto guard = slot_guard();
    return run_inference(guard.slot, prompt, opts, nullptr);
  }

  void generate(const std::string &prompt, const StreamCallback &callback) {
    generate(prompt, {}, callback);
  }

  void generate(const std::string &prompt, const GenerateOptions &opts,
                const StreamCallback &callback) {
    auto guard = slot_guard();
    run_inference(guard.slot, prompt, opts, callback);
  }

  std::vector<llama_token> tokenize(const std::string &text) const {
    auto vocab = llama_model_get_vocab(model_.get());
    return tokenize_text(vocab, text);
  }

  size_t token_count(const std::string &text) const {
    return tokenize(text).size();
  }

  inline ChatSession session(const std::string &system_prompt = "");

  void clear_cache() {
    std::lock_guard lock(mutex_);
    auto n = slots_.size();
    for (size_t i = 0; i < n; i++) {
      auto slot = std::move(slots_.front());
      slots_.pop();
      llama_memory_clear(llama_get_memory(slot.ctx.get()), true);
      slot.cached_tokens.clear();
      slots_.push(std::move(slot));
    }
  }

  std::string chat(const std::string &message,
                   const GenerateOptions &opts = {}) {
    return chat(std::vector<Message>{{"user", message}}, opts);
  }

  void chat(const std::string &message, const StreamCallback &callback) {
    chat(std::vector<Message>{{"user", message}}, callback);
  }

  void chat(const std::string &message, const GenerateOptions &opts,
            const StreamCallback &callback) {
    chat(std::vector<Message>{{"user", message}}, opts, callback);
  }

  std::string chat(const std::vector<Message> &messages,
                   const GenerateOptions &opts = {}) {
    auto formatted = apply_chat_template(messages);
    return generate(formatted, opts);
  }

  void chat(const std::vector<Message> &messages,
            const StreamCallback &callback) {
    chat(messages, {}, callback);
  }

  void chat(const std::vector<Message> &messages, const GenerateOptions &opts,
            const StreamCallback &callback) {
    auto formatted = apply_chat_template(messages);
    generate(formatted, opts, callback);
  }

private:
  struct Slot {
    llama_context_ptr ctx;
    llama_sampler_ptr smpl;
    std::vector<llama_token> cached_tokens;
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

  static std::vector<llama_token> tokenize_text(const llama_vocab *vocab,
                                                  const std::string &text) {
    std::vector<llama_token> tokens(text.size() + 16);
    auto n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(),
                            tokens.size(), true, true);
    if (n < 0) {
      tokens.resize(static_cast<size_t>(-n));
      n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(),
                          tokens.size(), true, true);
      if (n < 0) {
        throw std::runtime_error("Failed to tokenize text");
      }
    }
    tokens.resize(static_cast<size_t>(n));
    return tokens;
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
    // Falls back to raw concatenation if the model has no chat template
    auto tmpl = llama_model_chat_template(model_.get(), nullptr);
    if (!tmpl) { return concat_contents(messages); }

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
                            const GenerateOptions &opts,
                            const StreamCallback &callback) {
    llama_sampler_reset(slot.smpl.get());
    auto vocab = llama_model_get_vocab(model_.get());

    auto tokens = tokenize_text(vocab, prompt);

    auto n_ctx = llama_n_ctx(slot.ctx.get());
    auto max_tokens = opts.max_tokens;
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

    // KV cache reuse: find common prefix with cached tokens
    size_t prefix_len = 0;
    {
      auto limit = std::min(slot.cached_tokens.size(), tokens.size());
      while (prefix_len < limit &&
             slot.cached_tokens[prefix_len] == tokens[prefix_len]) {
        ++prefix_len;
      }
    }

    // Always re-decode at least the last prompt token to get fresh logits
    if (prefix_len > 0 && prefix_len == tokens.size()) {
      --prefix_len;
    }

    if (prefix_len == 0) {
      llama_memory_clear(llama_get_memory(slot.ctx.get()), true);
    } else {
      // Remove everything after prefix (stale prompt tokens + generated tokens)
      llama_memory_seq_rm(llama_get_memory(slot.ctx.get()), 0,
                          static_cast<llama_pos>(prefix_len), -1);
    }

    if (prefix_len < tokens.size()) {
      if (llama_decode(
              slot.ctx.get(),
              llama_batch_get_one(tokens.data() + prefix_len,
                                  tokens.size() - prefix_len))) {
        throw std::runtime_error("llama_decode failed on prompt");
      }
    }
    slot.cached_tokens = std::move(tokens);

    // Grammar sampler is inserted at position 0 (before temp/dist)
    bool has_grammar = !opts.grammar.empty();
    if (has_grammar) {
      auto root =
          opts.grammar_root.empty() ? "root" : opts.grammar_root.c_str();
      auto *smpl =
          llama_sampler_init_grammar(vocab, opts.grammar.c_str(), root);
      auto n_samplers = llama_sampler_chain_n(slot.smpl.get());
      std::vector<llama_sampler *> existing;
      for (int si = n_samplers - 1; si >= 0; --si) {
        existing.push_back(llama_sampler_chain_remove(slot.smpl.get(), si));
      }
      llama_sampler_chain_add(slot.smpl.get(), smpl);
      for (auto it = existing.rbegin(); it != existing.rend(); ++it) {
        llama_sampler_chain_add(slot.smpl.get(), *it);
      }
    }

    struct GrammarGuard {
      llama_sampler *chain;
      bool active;
      ~GrammarGuard() {
        if (active) {
          auto *removed = llama_sampler_chain_remove(chain, 0);
          llama_sampler_free(removed);
        }
      }
    } grammar_guard{slot.smpl.get(), has_grammar};

    // Compute the max length among stop sequences for buffer management
    size_t max_stop_len = 0;
    for (const auto &s : opts.stop) {
      if (s.size() > max_stop_len) max_stop_len = s.size();
    }
    bool has_stop = max_stop_len > 0;

    std::string result;
    std::string pending;  // Buffer for partial stop-sequence matching
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

      if (has_stop) {
        pending.append(sv);

        // Check for complete stop sequence match
        bool found_stop = false;
        for (const auto &s : opts.stop) {
          if (s.empty()) continue;
          auto pos = pending.find(s);
          if (pos != std::string::npos) {
            // Emit text before the stop sequence
            auto emit = std::string_view(pending.data(), pos);
            if (callback) {
              if (!emit.empty()) callback(emit);
            } else {
              result.append(emit);
            }
            found_stop = true;
            break;
          }
        }
        if (found_stop) { pending.clear(); break; }

        // Flush confirmed-safe bytes that can't be part of a stop sequence
        if (pending.size() > max_stop_len) {
          auto safe = pending.size() - max_stop_len;
          auto emit = std::string_view(pending.data(), safe);
          if (callback) {
            if (!callback(emit)) break;
          } else {
            result.append(emit);
          }
          pending.erase(0, safe);
        }
      } else {
        if (callback) {
          if (!callback(sv)) break;
        } else {
          result.append(sv);
        }
      }

      if (llama_decode(slot.ctx.get(), llama_batch_get_one(&new_token, 1))) {
        throw std::runtime_error("llama_decode failed during generation");
      }
    }

    // Flush remaining pending buffer (no stop sequence matched)
    if (!pending.empty()) {
      if (callback) {
        callback(pending);
      } else {
        result.append(pending);
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

  std::string say(const std::string &message,
                  const GenerateOptions &opts = {}) {
    return say_impl(message, [&] { return llm_.chat(history_, opts); });
  }

  void say(const std::string &message, const StreamCallback &callback) {
    say(message, {}, callback);
  }

  void say(const std::string &message, const GenerateOptions &opts,
           const StreamCallback &callback) {
    say_impl(message, [&] {
      std::string reply;
      llm_.chat(history_, opts, [&](std::string_view token) {
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

inline ChatSession Llama::session(const std::string &system_prompt) {
  return ChatSession(*this, system_prompt);
}

} // namespace llamalib

#endif // CPP_LLAMALIB_H
