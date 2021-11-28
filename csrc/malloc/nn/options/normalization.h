#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>
#include <vector>

namespace malloc {
namespace nn {

struct TORCH_API LayerNormOptions {
  /* implicit */ LayerNormOptions(std::vector<int64_t> normalized_shape);
  /// input shape from an expected input.
  TORCH_ARG(std::vector<int64_t>, normalized_shape);

  TORCH_ARG(double, eps) = 1e-5;

  TORCH_ARG(bool, elementwise_affine) = true;
};

namespace functional {

struct TORCH_API LayerNormFuncOptions {
  /* implicit */ LayerNormFuncOptions(std::vector<int64_t> normalized_shape);
  /// input shape from an expected input.
  TORCH_ARG(std::vector<int64_t>, normalized_shape);

  TORCH_ARG(Tensor, weight) = {};

  TORCH_ARG(Tensor, bias) = {};

  /// a value added to the denominator for numerical stability. ``Default: 1e-5``.
  TORCH_ARG(double, eps) = 1e-5;
};

} // namespace functional

} // namespace nn
} // namespace malloc
