#pragma once

#include "LowererUtils.hpp"

namespace modelica
{
	llvm::Expected<llvm::Value*> lowerExp(
			llvm::IRBuilder<>& builder,
			llvm::Module& mod,
			llvm::Function* fun,
			const SimExp& exp);

	llvm::Expected<llvm::AllocaInst*> lowerConstant(
			llvm::IRBuilder<>& builder, const SimExp& exp);
}	 // namespace modelica
