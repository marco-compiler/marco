#pragma once

#include "LowererUtils.hpp"

namespace modelica
{
	llvm::Expected<llvm::Value*> lowerExp(
			LowererContext& info, const ModExp& exp);

	llvm::Expected<llvm::AllocaInst*> lowerConstant(
			LowererContext& context, const ModExp& exp);
}	 // namespace modelica
