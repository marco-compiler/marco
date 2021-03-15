#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <modelica/frontend/Parser.hpp>
#include <modelica/frontend/Passes.h>
#include <modelica/frontend/SymbolTable.hpp>
#include <modelica/mlirlowerer/CodeGen.h>

using namespace llvm::cl;
using namespace modelica;
using namespace std;

static OptionCategory optionCategory("MLIRLowerer options");

static opt<string> inputFileName(Positional, desc("<input-file>"), init("-"), cat(optionCategory));
static opt<string> outputFile("o", desc("<output-file>"), init("-"), cat(optionCategory));

static llvm::ExitOnError exitOnErr;

int main(int argc, char* argv[])
{
	llvm::InitLLVM y(argc, argv);
	ParseCommandLineOptions(argc, argv);

	auto errorOrBuffer = llvm::MemoryBuffer::getFileOrSTDIN(inputFileName);
	error_code error;
	llvm::raw_fd_ostream os(outputFile, error, llvm::sys::fs::F_None);

	if (error)
	{
		llvm::errs() << error.message();
		return -1;
	}

	auto buffer = exitOnErr(errorOrToExpected(move(errorOrBuffer)));
	Parser parser(buffer->getBufferStart());
	auto ast = exitOnErr(parser.classDefinition());

	PassManager passManager;
	passManager.addPass(createTypeCheckingPass());
	passManager.addPass(createConstantFolderPass());
	passManager.addPass(createBreakRemovingPass());
	passManager.addPass(createReturnRemovingPass());
	exitOnErr(passManager.run(ast));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);
	auto module = lowerer.lower(ast);

	if (!module)
	{
		llvm::errs() << "Failed to convert module\n";
		return -1;
	}

	if (mlir::failed(convertToLLVMDialect(&context, *module)))
	{
		llvm::errs() << "Failed to convert module to LLVM\n";
		return -1;
	}

	// Register the conversions to LLVM IR
	mlir::registerLLVMDialectTranslation(*module->getContext());
	mlir::registerOpenMPDialectTranslation(*module->getContext());

	// Convert to LLVM IR
	llvm::LLVMContext llvmContext;
	auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);

	if (!llvmModule) {
		llvm::errs() << "Failed to emit LLVM IR\n";
		return -1;
	}

	// Initialize LLVM targets
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	// Optimize the IR
	auto optPipeline = mlir::makeOptimizingTransformer(3, 0, nullptr);

	if (auto err = optPipeline(llvmModule.get())) {
		llvm::errs() << "Failed to optimize LLVM IR: " << err << "\n";
		return -1;
	}

	llvm::errs() << *llvmModule << "\n";
	llvm::WriteBitcodeToFile(*llvmModule, os);
	return 0;
}
