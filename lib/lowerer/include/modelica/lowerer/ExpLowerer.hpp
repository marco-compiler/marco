#pragma once

#include "marco/lowerer/LowererUtils.hpp"

namespace marco
{
	llvm::Expected<llvm::Value*> lowerExp(
			LowererContext& info, const ModExp& exp);

	llvm::Expected<llvm::AllocaInst*> lowerConstant(
			LowererContext& context, const ModExp& exp);

	llvm::Expected<llvm::Value*> castReturnValue(
			LowererContext& info, llvm::Value* val, const ModType& type);
}	 // namespace marco
