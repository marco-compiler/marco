#pragma once

#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"
#include "modelica/lowerer/LowererUtils.hpp"
#include "modelica/model/ModBltBlock.hpp"

namespace modelica
{
	llvm::Expected<llvm::SmallVector<llvm::Value*, 3>> lowerBltBlock(
			LowererContext& info, const ModBltBlock& bltBlock);

}	 // namespace modelica
