#ifndef MARCO_AST_PASSMANAGER_H
#define MARCO_AST_PASSMANAGER_H

#include "marco/AST/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace marco::ast
{
	class Class;

	class PassManager
	{
		public:
      void addPass(std::unique_ptr<Pass> pass);
      bool run(std::unique_ptr<Class>& cls);

		private:
      llvm::SmallVector<std::unique_ptr<Pass>, 3> passes;
	};
}

#endif // MARCO_AST_PASSMANAGER_H
