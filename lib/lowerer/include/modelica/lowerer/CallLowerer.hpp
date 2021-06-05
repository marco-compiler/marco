#pragma once

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"
#include "marco/lowerer/LowererUtils.hpp"
#include "marco/model/ModCall.hpp"

namespace marco
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

}	 // namespace marco
