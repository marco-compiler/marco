#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/Utils.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>
#include <modelica/frontend/Parser.hpp>
#include <modelica/frontend/Passes.h>
#include <modelica/mlirlowerer/CodeGen.h>

using namespace llvm;
using namespace modelica;
using namespace std;

static cl::OptionCategory optionCategory("MLIR lowerer options");

static cl::opt<string> inputFile(cl::Positional, cl::desc("<input-file>"), cl::init("-"), cl::cat(optionCategory));
static cl::opt<string> outputFile("o", cl::desc("<output-file>"), cl::init("-"), cl::cat(optionCategory));
static cl::opt<bool> x86("32", cl::desc("Use 32-bit values instead of 64-bit ones"), cl::init(false), cl::cat(optionCategory));
static cl::opt<bool> openmp("omp", cl::desc("Enable OpenMP usage"), cl::init(false), cl::cat(optionCategory));

enum OptLevel {
	O0, O1, O2, O3
};

static cl::opt<OptLevel> optimizationLevel(cl::desc("Optimization level:"),
																					 cl::values(
																							 clEnumValN(O0, "O0", "No optimizations"),
																							 clEnumValN(O1, "O1", "Trivial optimizations"),
																							 clEnumValN(O2, "O2", "Default optimizations"),
																							 clEnumValN(O3, "O3", "Expensive optimizations")),
																					 cl::cat(optionCategory),
																					 cl::init(O2));

static cl::opt<bool> printParsedAST("print-parsed-ast", cl::desc("Print the AST right after being parsed"), cl::init(false), cl::cat(optionCategory));
static cl::opt<bool> printLegalizedAST("print-legalized-ast", cl::desc("Print the AST after it has been legalized"), cl::init(false), cl::cat(optionCategory));
static cl::opt<bool> printModelicaDialectIR("print-modelica", cl::desc("Print the Modelica dialect IR"), cl::init(false), cl::cat(optionCategory));
static cl::opt<bool> printLLVMDialectIR("print-llvm", cl::desc("Print the LLVM dialect IR"), cl::init(false), cl::cat(optionCategory));

static cl::opt<bool> debug("d", cl::desc("Keep debug information in the final IR"), cl::init(false), cl::cat(optionCategory));

static llvm::ExitOnError exitOnErr;

int main(int argc, char* argv[])
{
	HideUnrelatedOptions(optionCategory);
	cl::ParseCommandLineOptions(argc, argv);

	auto errorOrBuffer = llvm::MemoryBuffer::getFileOrSTDIN(inputFile);
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

	if (printParsedAST)
		ast.dump();

	// Run frontend passes
	modelica::PassManager frontendPassManager;
	frontendPassManager.addPass(createTypeCheckingPass());
	frontendPassManager.addPass(createConstantFolderPass());
	frontendPassManager.addPass(createBreakRemovingPass());
	frontendPassManager.addPass(createReturnRemovingPass());
	exitOnErr(frontendPassManager.run(ast));

	if (printLegalizedAST)
		ast.dump();

	// Create the MLIR module
	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = !x86.getValue();

	MLIRLowerer lowerer(context, modelicaOptions);
	auto module = lowerer.lower(ast);

	if (!module)
	{
		llvm::errs() << "Failed to emit Modelica IR\n";
		return -1;
	}

	if (printModelicaDialectIR)
		module->dump();

	// Convert to LLVM dialect
	modelica::ModelicaConversionOptions conversionOptions;
	conversionOptions.openmp = openmp;
	conversionOptions.debug = debug;

	if (mlir::failed(lowerer.convertToLLVMDialect(*module, conversionOptions)))
	{
		llvm::errs() << "Failed to convert module to LLVM\n";
		return -1;
	}

	if (printLLVMDialectIR)
		module->dump();

	// Register the conversions to LLVM IR
	mlir::registerLLVMDialectTranslation(*module->getContext());
	mlir::registerOpenMPDialectTranslation(*module->getContext());

	// Initialize LLVM targets
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	// Convert to LLVM IR
	llvm::LLVMContext llvmContext;
	auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);

	if (!llvmModule) {
		llvm::errs() << "Failed to emit LLVM IR\n";
		return -1;
	}

	// Optimize the IR
	int optLevel = 0;

	if (optimizationLevel == O0)
		optLevel = 0;
	else if (optimizationLevel == O1)
		optLevel = 1;
	else if (optimizationLevel == O2)
		optLevel = 2;
	else if (optimizationLevel == O3)
		optLevel = 3;

	auto optPipeline = mlir::makeOptimizingTransformer(optLevel, 0, nullptr);

	if (auto err = optPipeline(llvmModule.get())) {
		llvm::errs() << "Failed to optimize LLVM IR: " << err << "\n";
		return -1;
	}

	llvm::errs() << *llvmModule << "\n";

	llvm::WriteBitcodeToFile(*llvmModule, os);
	return 0;
}
