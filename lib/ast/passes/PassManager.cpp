#include "marco/ast/PassManager.h"

using namespace marco::ast;

void PassManager::addPass(std::unique_ptr<Pass> pass)
{
	passes.push_back(std::move(pass));
}

llvm::Error PassManager::run(llvm::ArrayRef<std::unique_ptr<Class>> classes)
{
	for (auto& pass : passes)
		if (auto error = pass->run(classes); error)
			return error;

	return llvm::Error::success();
}
