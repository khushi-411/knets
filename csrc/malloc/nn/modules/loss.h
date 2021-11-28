#pragma once

#include <torch/expanding_array.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/functional/loss.h>
#include <torch/nn/options/loss.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstddef>
#include <vector>

namespace malloc {
namespace nn {

struct TORCH_API L1LossImpl : Cloneable<L1LossImpl> {
  explicit L1LossImpl(const L1LossOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `L1Loss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  L1LossOptions options;
};

TORCH_MODULE(L1Loss);

} // namespace nn
} // namespace malloc
