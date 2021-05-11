#include <modelica/frontend/PassManager.h>

using namespace modelica::frontend;

void PassManager::addPass(std::unique_ptr<Pass> pass)
{
	passes.push_back(std::move(pass));
}

llvm::Error PassManager::run(Class& cls)
{
	for (auto& pass : passes)
		if (auto error = pass->run(cls); error)
			return error;

	return llvm::Error::success();
}
