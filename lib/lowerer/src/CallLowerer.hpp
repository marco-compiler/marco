#pragma once

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"
#include "modelica/lowerer/LowererUtils.hpp"
#include "modelica/model/ModCall.hpp"

namespace modelica
{
	llvm::Expected<llvm::Value*> lowerCall(
			LowererContext& info, const ModCall& call);

	llvm::Expected<llvm::Value*> lowerCall(
			llvm::Value* outLocation, LowererContext& info, const ModCall& call);

	llvm::Expected<llvm::Value*> invoke(
			LowererContext& info,
			llvm::StringRef name,
			llvm::ArrayRef<llvm::Value*> args,
			llvm::Type* returnType = nullptr);

}	 // namespace modelica
