// Reference: https://github.com/khushi-411/pytorch/blob/master/torch/csrc/api/include/torch/nn/modules/container/functional.h

#pragma once

#include <torch/csrc/utils/variadic.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <functional>
#include <utility>

namespace malloc {
namespace nn {

class KHUSHI_API FunctionalImpl : public torch::nn::Cloneable<FunctionalImpl> {
 public:
  using Function = std::function<Tensor(Tensor)>;

  /// Constructs a `Functional` from a function object.
  explicit FunctionalImpl(Function function);

  template <
      typename SomeFunction,
      typename... Args,
      typename = torch::enable_if_t<(sizeof...(Args) > 0)>>
  explicit FunctionalImpl(SomeFunction original_function, Args&&... args)
  // NOLINTNEXTLINE(modernize-avoid-bind)
      : function_(std::bind(
            original_function,
            /*input=*/std::placeholders::_1,
            std::forward<Args>(args)...)) {
}

void reset() override;

  /// Pretty prints the `Functional` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Forwards the `input` tensor to the underlying (bound) function object.
  Tensor forward(Tensor input);

  /// Calls forward(input).
  Tensor operator()(Tensor input);

  bool is_serializable() const override;

 private:
  Function function_;
};

TORCH_MODULE(Functional);

} // namespace nn
} // namespace malloc
