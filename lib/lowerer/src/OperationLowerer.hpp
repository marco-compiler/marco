#pragma once

#include "LowererUtils.hpp"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"
#include "modelica/model/ModExp.hpp"

namespace modelica
{
	template<ModExpKind kind, typename... Values>
	llvm::Value* op(llvm::IRBuilder<>& builder, Values... arg1);
	llvm::Expected<llvm::Value*> lowerAtOperation(
			LoweringInfo& info, const ModExp& exp, bool loadOld);
	llvm::Expected<llvm::Value*> lowerNegate(
			LoweringInfo& info, const ModExp& arg1, bool loadOld);
	llvm::Expected<llvm::Value*> lowerInduction(
			LoweringInfo& info, const ModExp& arg1, bool loadOld);

}	 // namespace modelica
