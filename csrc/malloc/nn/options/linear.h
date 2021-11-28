#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>
#include <c10/util/variant.h>

namespace malloc {
namespace nn {

struct TORCH_API LinearOptions {
  LinearOptions(int64_t in_features, int64_t out_features);
  /// size of each input sample
  TORCH_ARG(int64_t, in_features);

  /// size of each output sample
  TORCH_ARG(int64_t, out_features);

  /// If set to false, the layer will not learn an additive bias. Default: true
  TORCH_ARG(bool, bias) = true;
};

} // namespace nn
} // namespace malloc
