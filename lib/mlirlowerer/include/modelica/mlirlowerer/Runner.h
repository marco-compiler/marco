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
		Runner(mlir::MLIRContext* context, mlir::ModuleOp module)
				: context(context), module(std::move(module))
		{
		}

		template<typename... T>
		mlir::LogicalResult run(llvm::StringRef function, T&... params)
		{
			// Initialize LLVM targets
			llvm::InitializeNativeTarget();
			llvm::InitializeNativeTargetAsmPrinter();

			// Create the engine to run the code
			llvm::SmallVector<llvm::StringRef, 3> libraries;
			libraries.push_back("/opt/llvm/lib/libmlir_runner_utils.so");
			libraries.push_back("/opt/llvm/lib/libmlir_c_runner_utils.so");
			//libraries.push_back("/mnt/d/modelica/cmake-build-gcc-debug/lib/runtime/libruntime-d.so");

			//mlir::registerLLVMDialectTranslation(*module->getContext());

			auto maybeEngine = mlir::ExecutionEngine::create(module, nullptr, {}, llvm::None, libraries);

			if (!maybeEngine)
			{
				llvm::errs() << "Failed to create the engine\n";
				return mlir::failure();
			}

			auto& engine = maybeEngine.get();

			// Run
			llvm::SmallVector<void*, 3> args = { ((void*) &params)... };

			if (engine->invoke(function, args))
			{
				llvm::errs() << "JIT invocation failed\n";
				return mlir::failure();
			}

			return mlir::success();
		}

		private:
		mlir::MLIRContext* context;
		mlir::ModuleOp module;
	};
}
