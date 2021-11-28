#pragma once

#include <string>

#include <ATen/core/Reduction.h>
#include <c10/util/Exception.h>
#include <c10/util/variant.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#define TORCH_ENUM_DECLARE(name) \
namespace torch { \
namespace enumtype { \


