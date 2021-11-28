#include <malloc/nn/modules/activation.h>
#include <malloc/nn/functional/activation.h>
#include <malloc/nn/init.h>

namespace F = torch::nn::functional;

namespace malloc {
namespace nn {

ReLUImpl::ReLUImpl(const ReLUOptions& options_) : options(options_) {}

Tensor ReLUImpl::forward(Tensor input) {
  return F::detail::relu(input, options.inplace());
}

void ReLUImpl::reset() {}

void ReLUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ReLU(";
  if (options.inplace()) {
    stream << std::boolalpha << "inplace=" << options.inplace();
  }
  stream << ")";
}

} // namespace nn
} // namespace malloc
