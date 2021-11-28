#pragma once

#include <torch/arg.h>
#include <malloc/enum.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace malloc {
namespace nn {

// Options for MSELoss module.
struct TORCH_API MSELossOptions {
  typedef c10::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

  TORCH_OPTIONS_CTOR_VARIANT_ARG3(MSELossOptions, reduction, kNone, kMean, kSum)

  /// Specifies the reduction to apply to the output.
  /// ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {

using MSELossFuncOptions = MSELossOptions;
} // namespace functional

} // namespace nn
} // namespace malloc
