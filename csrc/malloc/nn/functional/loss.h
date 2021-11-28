#pragma once

#include <ATen/ExpandUtils.h>
#include <torch/nn/functional/activation.h>
#include <torch/nn/options/loss.h>

// TODO: Why to take from three different places? Which is better?
namespace malloc {
namespace nn {
namespace functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor l1_loss(
    const Tensor& input,
    const Tensor& target,
    L1LossFuncOptions::reduction_t reduction) {
  return torch::l1_loss(
    input,
    target,
    enumtype::reduction_get_enum(reduction));
}
} // namespace detail
#endif

inline Tensor l1_loss(
    const Tensor& input,
    const Tensor& target,
    const L1LossFuncOptions& options = {}) {
  return detail::l1_loss(input, target, options.reduction());
}

} // namespace functional
} // namespace nn
} // namespace malloc
