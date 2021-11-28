#include <malloc/nn/modules/normalization.h>

#include <torch/cuda.h>
#include <torch/utils.h>
#include <torch/nn/init.h>

#include <ostream>
#include <utility>

namespace F = torch::nn::functional;
