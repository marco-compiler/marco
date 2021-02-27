#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Target/LLVMIR.h>

namespace modelica
{
	class Runner
	{
		public:
		template<typename T> using Result = mlir::ExecutionEngine::Result<T>;

		Runner(mlir::ModuleOp module, llvm::ArrayRef<mlir::StringRef> libraries = {}, unsigned int speedOptimization = 0, unsigned int sizeOptimization = 0);

		template <typename T>
		static Result<T> result(T &t) {
			return Result<T>(t);
		}

		template<typename... T>
		mlir::LogicalResult run(llvm::StringRef function, T... args)
		{
			if (!initialized)
			{
				llvm::errs() << "Engine not initialized\n";
				return mlir::failure();
			}

			if (engine->invoke(function, args...))
			{
				llvm::errs() << "JIT invocation failed\n";
				return mlir::failure();
			}

			return mlir::success();
		}

		private:
		bool initialized;
		mlir::ModuleOp module;
		std::unique_ptr<mlir::ExecutionEngine> engine;
	};
}
