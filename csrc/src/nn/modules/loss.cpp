#include <malloc/nn/modules/loss.h>

namespace F = torch::nn::functional;

namespace malloc {
namespace nn {

MSELossImpl::MSELossImpl(const MSELossOptions& options_) : options(options_) {}

void MSELossImpl::reset() {}

void MSELossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::MSELoss()";
}

Tensor MSELossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::detail::mse_loss(input, target, options.reduction());
}

} // namespace nn
} // namespace malloc
