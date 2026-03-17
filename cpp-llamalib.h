#ifndef CPP_LLAMALIB_H
#define CPP_LLAMALIB_H

#include "llama.h"
#include "llama-cpp.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
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

struct Params {
  int n_gpu_layers = 99;
  int n_ctx = 2048;
  float temperature = 0.3f;
  int max_tokens = 512;
  int n_slots = 1;
};

class LLM {
public:
  LLM(const std::string &model_path, const Params &params = {})
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
      llama_sampler_chain_add(smpl.get(),
                              llama_sampler_init_temp(params.temperature));
      llama_sampler_chain_add(smpl.get(),
                              llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

      slots_.push({std::move(ctx), std::move(smpl)});
    }
  }

  ~LLM() {
    if (model_) { detail::backend_free(); }
  }

  LLM(const LLM &) = delete;
  LLM &operator=(const LLM &) = delete;

  LLM(LLM &&other) noexcept
      : params_(other.params_),
        model_(std::move(other.model_)),
        slots_(std::move(other.slots_)) {}

  LLM &operator=(LLM &&other) noexcept {
    if (this != &other) {
      // Destroy slots before model (contexts reference the model).
      // No backend_free needed: this is ownership transfer, not destruction.
      // other's destructor will skip backend_free since other.model_ is null.
      slots_ = {};
      model_.reset();
      params_ = other.params_;
      model_ = std::move(other.model_);
      slots_ = std::move(other.slots_);
    }
    return *this;
  }

  std::string generate(const std::string &prompt) {
    return generate(prompt, params_.max_tokens);
  }

  std::string generate(const std::string &prompt, int max_tokens) {
    auto slot = acquire_slot();
    auto result = run_inference(slot, prompt, max_tokens);
    release_slot(std::move(slot));
    return result;
  }

private:
  struct Slot {
    llama_context_ptr ctx;
    llama_sampler_ptr smpl;
  };

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

  std::string run_inference(Slot &slot, const std::string &prompt,
                            int max_tokens) {
    auto vocab = llama_model_get_vocab(model_.get());

    std::vector<llama_token> tokens(prompt.size() + 16);
    auto n = llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                            tokens.data(), tokens.size(), true, true);
    tokens.resize(n);

    llama_kv_self_clear(slot.ctx.get());
    llama_decode(slot.ctx.get(),
                 llama_batch_get_one(tokens.data(), tokens.size()));

    std::string result;
    for (auto i = 0; i < max_tokens; i++) {
      auto new_token =
          llama_sampler_sample(slot.smpl.get(), slot.ctx.get(), -1);
      if (llama_vocab_is_eog(vocab, new_token)) break;

      char buf[256];
      auto len =
          llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
      if (len > 0) result.append(buf, len);

      llama_decode(slot.ctx.get(), llama_batch_get_one(&new_token, 1));
    }
    return result;
  }

  Params params_;
  llama_model_ptr model_;
  std::queue<Slot> slots_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

} // namespace llamalib

#endif // CPP_LLAMALIB_H
