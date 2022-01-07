#include <llvm/Support/TargetSelect.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h>
#include <marco/codegen/Runner.h>

using namespace marco::jit;

Runner::Runner(mlir::ModuleOp module, llvm::ArrayRef<mlir::StringRef> libraries, unsigned int speedOptimization, unsigned int sizeOptimization)
		: module(std::move(module))
{
	// Register the conversions to LLVM IR
	mlir::registerLLVMDialectTranslation(*module->getContext());
	mlir::registerOpenMPDialectTranslation(*module->getContext());

	// Initialize LLVM targets
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	llvm::SmallVector<mlir::StringRef, 3> allLibraries(libraries.begin(), libraries.end());
	//allLibraries.push_back(RUNTIME_LIBRARY);

	// Create the engine
	auto optPipeline = mlir::makeOptimizingTransformer(speedOptimization, sizeOptimization, nullptr);
	auto maybeEngine = mlir::ExecutionEngine::create(module, nullptr, optPipeline, llvm::None, allLibraries);

	if (!maybeEngine)
	{
		llvm::errs() << "Failed to create the engine\n";
		initialized = false;
	}
	else
	{
		engine = std::move(maybeEngine.get());
		initialized = true;
	}
}
