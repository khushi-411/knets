#pragma once

// TODO: Write correct imports
#include <torch/nn/cloneable.h>
#include <torch/nn/options/activation.h>
#include <torch/nn/functional/activation.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/linear.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace malloc {
namespace nn {

class KHUSHI_API ReLUImpl : public torch::nn::Cloneable<ReLUImpl> {
 public:
  explicit ReLUImpl(const ReLUOptions& options_ = {});

  Tensor forward(Tensor input);

  void reset() override;

  /// Pretty prints the `ReLU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ReLUOptions options;
};

TORCH_MODULE(ReLU);

} // namespace nn
} // namespace malloc
