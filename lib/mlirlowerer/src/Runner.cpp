#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h>
#include <modelica/mlirlowerer/Runner.h>

using namespace modelica;

Runner::Runner(mlir::ModuleOp module, llvm::ArrayRef<mlir::StringRef> libraries, unsigned int speedOptimization, unsigned int sizeOptimization)
		: module(std::move(module))
{
	// Register the conversion to LLVM IR
	mlir::registerLLVMDialectTranslation(*module->getContext());
	mlir::registerOpenMPDialectTranslation(*module->getContext());

	// Initialize LLVM targets
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	//libs.push_back("/opt/llvm/lib/libmlir_runner_utils.so");
	//libs.push_back("/opt/llvm/lib/libmlir_c_runner_utils.so");
	//libraries.push_back("/mnt/d/modelica/cmake-build-gcc-debug/lib/runtime/libruntime-d.so");

	// Create the engine
	auto optPipeline = mlir::makeOptimizingTransformer(speedOptimization, sizeOptimization, nullptr);
	auto maybeEngine = mlir::ExecutionEngine::create(module, nullptr, optPipeline, llvm::None, libraries);

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
