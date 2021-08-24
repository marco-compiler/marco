#pragma once

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"
#include "marco/lowerer/LowererUtils.hpp"
#include "marco/model/ModExp.hpp"

namespace marco
{
	template<ModExpKind kind, typename... Values>
	llvm::Expected<llvm::Value*> op(LowererContext& info, Values... arg1);
	llvm::Expected<llvm::Value*> lowerAtOperation(
			LowererContext& info, const ModExp& exp);
	llvm::Expected<llvm::Value*> lowerNegate(
			LowererContext& info, const ModExp& arg1);
	llvm::Expected<llvm::Value*> lowerInduction(
			LowererContext& info, const ModExp& arg1);

}	 // namespace marco
