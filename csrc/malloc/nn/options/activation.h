#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace malloc {
namespace nn {

struct KHUSHI_API ReLUOptions {
	// type of function: IMPLICIT; Why?
	ReLUOptions(bool inplace = false);

	/// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace);
};

namespace functional {

using ReLUFuncOptions = ReLUOptions;
} // namespace functional

} // nn
} // malloc
