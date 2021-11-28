#pragma once

#include <torch/nn/cloneable.h>
#include <malloc/nn/module.h>
#include <torch/nn/options/linear.h>
#include <torch/nn/pimpl.h>
#include <torch/nn/functional/linear.h>
#include <torch/types.h>

#include <cstddef>
#include <vector>

namespace malloc {
namespace nn {

class KHUSHI_API LinearImpl : public Cloneable<LinearImpl> {
	public:
		// TODO & NOTE: We used contructor over-loading; Why?
		// https://github.com/khushi-411/pytorch/blob/cd51d2a3ecc8ac579bee910f6bafe41a4c41ca80/torch/csrc/api/include/torch/enum.h#L14
		LinearImpl(int64_t in_features, int64_t out_features)
			: LinearImpl(LinearOptions(in_features, out_features)) {}
		explicit LinearImpl(const LinearOptions& options_);

		void reset() override;

		void reset_parameters();

		void pretty_print(std::ostresm& stream) const override;

		Tensor forward(const Tensor& input);

		LinearOptions options;

		Tensor weight;

		Tensor bias;
};

TORCH_MODULE(Linear);

} // namespace nn
} // namespace malloc
