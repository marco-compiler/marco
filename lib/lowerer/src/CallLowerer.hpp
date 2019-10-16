#pragma once

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"
#include "modelica/model/ModCall.hpp"

namespace modelica
{
	llvm::Expected<llvm::Value*> lowerCall(
			llvm::IRBuilder<>& builder,
			llvm::Module& module,
			llvm::Function* fun,
			const ModCall& call);
}
