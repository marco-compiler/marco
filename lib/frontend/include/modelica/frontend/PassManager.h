#pragma once

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <memory>

#include "Pass.h"

namespace modelica::frontend
{
	class Class;

	class PassManager
	{
		public:
		void addPass(std::unique_ptr<Pass> pass);
		llvm::Error run(Class& cls);

		private:
		llvm::SmallVector<std::unique_ptr<Pass>, 3> passes;
	};
}
