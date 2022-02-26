#ifndef MARCO_AST_PASSMANAGER_H
#define MARCO_AST_PASSMANAGER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "marco/AST/Pass.h"
#include <memory>

namespace marco::ast
{
	class Class;

	class PassManager
	{
		public:
		void addPass(std::unique_ptr<Pass> pass);
		llvm::Error run(llvm::ArrayRef<std::unique_ptr<Class>> classes);

		private:
		llvm::SmallVector<std::unique_ptr<Pass>, 3> passes;
	};
}

#endif // MARCO_AST_PASSMANAGER_H
