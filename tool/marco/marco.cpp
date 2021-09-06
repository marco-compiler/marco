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
#include <marco/frontend/Parser.h>
#include <marco/frontend/Passes.h>
#include <marco/mlirlowerer/CodeGen.h>

using namespace llvm;
using namespace marco;
using namespace std;

static cl::OptionCategory modelSolvingOptions("Model solving options");

static cl::opt<int> matchingMaxIterations("matching-max-iterations", cl::desc("Maximum number of iterations for the matching phase (default: 1000)"), cl::init(1000), cl::cat(modelSolvingOptions));
static cl::opt<int> sccMaxIterations("scc-max-iterations", cl::desc("Maximum number of iterations for the SCC resolution phase (default: 1000)"), cl::init(1000), cl::cat(modelSolvingOptions));
static cl::opt<codegen::Solver> solver(cl::desc("Solvers:"),
																			 cl::values(
																					 clEnumValN(codegen::ForwardEuler, "forward-euler", "Forward Euler (default)"),
																					 clEnumValN(codegen::CleverDAE, "clever-dae", "Clever DAE")),
																			 cl::init(codegen::ForwardEuler),
																			 cl::cat(modelSolvingOptions));

static cl::OptionCategory codeGenOptions("Code generation options");

static cl::list<std::string> inputFiles(cl::Positional, cl::desc("<input files>"), cl::OneOrMore, cl::cat(codeGenOptions));
static cl::opt<string> outputFile("o", cl::desc("<output-file>"), cl::init("-"), cl::cat(codeGenOptions));
static cl::opt<bool> emitMain("emit-main", cl::desc("Whether to emit the main function that will start the simulation (default: true)"), cl::init(true), cl::cat(codeGenOptions));
static cl::opt<bool> x86("32", cl::desc("Use 32-bit values instead of 64-bit ones"), cl::init(false), cl::cat(codeGenOptions));
static cl::opt<bool> inlining("no-inlining", cl::desc("Disable the inlining pass"), cl::init(false), cl::cat(codeGenOptions));
static cl::opt<bool> resultBuffersToArgs("no-result-buffers-to-args", cl::desc("Don't move the static output buffer to input arguments"), cl::init(false), cl::cat(codeGenOptions));
static cl::opt<bool> cse("no-cse", cl::desc("Disable CSE pass"), cl::init(false), cl::cat(codeGenOptions));
static cl::opt<bool> openmp("omp", cl::desc("Enable OpenMP usage"), cl::init(false), cl::cat(codeGenOptions));
static cl::opt<bool> disableRuntimeLibrary("disable-runtime-library", cl::desc("Avoid the calls to the external runtime library functions (only when a native implementation of the operation exists)"), cl::init(false), cl::cat(codeGenOptions));
static cl::opt<bool> emitCWrappers("emit-c-wrappers", cl::desc("Emit C wrappers"), cl::init(false), cl::cat(codeGenOptions));

enum OptLevel {
	O0, O1, O2, O3
};

static cl::opt<OptLevel> optimizationLevel(cl::desc("Optimization level:"),
																					 cl::values(
																							 clEnumValN(O0, "O0", "No optimizations"),
																							 clEnumValN(O1, "O1", "Trivial optimizations"),
																							 clEnumValN(O2, "O2", "Default optimizations"),
																							 clEnumValN(O3, "O3", "Expensive optimizations")),
																					 cl::cat(codeGenOptions),
																					 cl::init(O2));

static cl::OptionCategory debugOptions("Debug options");

static cl::opt<bool> printParsedAST("print-parsed-ast", cl::desc("Print the AST right after being parsed"), cl::init(false), cl::cat(debugOptions));
static cl::opt<bool> printLegalizedAST("print-legalized-ast", cl::desc("Print the AST after it has been legalized"), cl::init(false), cl::cat(debugOptions));
static cl::opt<bool> printModelicaDialectIR("print-modelica", cl::desc("Print the Modelica dialect IR obtained right after the AST lowering"), cl::init(false), cl::cat(debugOptions));
static cl::opt<bool> printLLVMDialectIR("print-llvm", cl::desc("Print the LLVM dialect IR"), cl::init(false), cl::cat(debugOptions));

static cl::opt<bool> debug("d", cl::desc("Keep debug information in the final IR"), cl::init(false), cl::cat(debugOptions));
static cl::opt<bool> enableAssertions("enable-assertions", cl::desc("Enable assertions (default: true for O0, false otherwise)"), cl::init(true), cl::cat(debugOptions));

static cl::OptionCategory simulationOptions("Simulation options");

static cl::opt<double> startTime("start-time", cl::desc("Start time (in seconds) (default: 0)"), cl::init(0), cl::cat(simulationOptions));
static cl::opt<double> endTime("end-time", cl::desc("End time (in seconds) (default: 10)"), cl::init(10), cl::cat(simulationOptions));
static cl::opt<double> timeStep("time-step", cl::desc("Time step (in seconds) (default: 0.1)"), cl::init(0.1), cl::cat(simulationOptions));

static llvm::ExitOnError exitOnErr;

int main(int argc, char* argv[])
{
	//get command line options
	llvm::SmallVector<const cl::OptionCategory*> categories;
	categories.push_back(&modelSolvingOptions);
	categories.push_back(&codeGenOptions);
	categories.push_back(&debugOptions);
	categories.push_back(&simulationOptions);
	HideUnrelatedOptions(categories);

	cl::ParseCommandLineOptions(argc, argv);

	error_code error;
	llvm::raw_fd_ostream os(outputFile, error, llvm::sys::fs::F_None);

	VariableFilter vf = VariableFilter();

	std::string test = "x1[1:2]";
	VariableFilterParser parser(test);
	parser.parseExpressionElement(vf);


	if (error)
	{
		llvm::errs() << error.message();
		return -1;
	}

	llvm::SmallVector<std::unique_ptr<frontend::Class>, 3> classes;

	if (inputFiles.empty())
		inputFiles.addValue("-");

	for (const auto& inputFile : inputFiles)
	{
		auto errorOrBuffer = llvm::MemoryBuffer::getFileOrSTDIN(inputFile);
		auto buffer = exitOnErr(errorOrToExpected(move(errorOrBuffer)));
		frontend::Parser parser(inputFile, buffer->getBufferStart());
		auto ast = exitOnErr(parser.classDefinition());
		classes.push_back(std::move(ast));
	}

	if (printParsedAST)
	{
		for (const auto& cls : classes)
			cls->dump(os);

		return 0;
	}

	// Run frontend passes
	frontend::PassManager frontendPassManager;
	frontendPassManager.addPass(frontend::createTypeCheckingPass());
	frontendPassManager.addPass(frontend::createConstantFolderPass());
	exitOnErr(frontendPassManager.run(classes));

	if (printLegalizedAST)
	{
		for (const auto& cls : classes)
			cls->dump(os);

		return 0;
	}

	// Create the MLIR module
	mlir::MLIRContext context;

	codegen::ModelicaOptions modelicaOptions;
	modelicaOptions.startTime = startTime;
	modelicaOptions.endTime = endTime;
	modelicaOptions.timeStep = timeStep;

	codegen::MLIRLowerer lowerer(context, modelicaOptions);
	auto module = lowerer.run(classes);

	if (!module)
	{
		llvm::errs() << "Failed to emit Modelica IR\n";
		return -1;
	}

	if (printModelicaDialectIR)
	{
		os << *module;
		return 0;
	}

	// Convert to LLVM dialect
	codegen::ModelicaLoweringOptions loweringOptions;
	loweringOptions.solveModelOptions.emitMain = emitMain;
	loweringOptions.solveModelOptions.variableFilter = &vf;
	loweringOptions.solveModelOptions.matchingMaxIterations = matchingMaxIterations;
	loweringOptions.solveModelOptions.sccMaxIterations = sccMaxIterations;
	loweringOptions.solveModelOptions.solver = solver;
	loweringOptions.inlining = !inlining;
	loweringOptions.resultBuffersToArgs = !resultBuffersToArgs;
	loweringOptions.cse = !cse;
	loweringOptions.openmp = openmp;
	loweringOptions.x64 = !x86.getValue();
	loweringOptions.conversionOptions.useRuntimeLibrary = !disableRuntimeLibrary;
	loweringOptions.llvmOptions.emitCWrappers = emitCWrappers;
	loweringOptions.debug = debug;

	if (optimizationLevel == O0)
	{
		loweringOptions.functionsVectorizationOptions.assertions = enableAssertions;
		loweringOptions.conversionOptions.assertions = enableAssertions;
		loweringOptions.llvmOptions.assertions = enableAssertions;
	}
	else
	{
		if (enableAssertions.getNumOccurrences() == 0)
		{
			loweringOptions.functionsVectorizationOptions.assertions = false;
			loweringOptions.conversionOptions.assertions = false;
			loweringOptions.llvmOptions.assertions = false;
		}
		else
		{
			loweringOptions.functionsVectorizationOptions.assertions = enableAssertions;
			loweringOptions.conversionOptions.assertions = enableAssertions;
			loweringOptions.llvmOptions.assertions = enableAssertions;
		}
	}

	if (mlir::failed(lowerer.convertToLLVMDialect(*module, loweringOptions)))
	{
		llvm::errs() << "Failed to convert module to LLVM\n";
		return -1;
	}

	if (printLLVMDialectIR)
	{
		os << *module;
		return 0;
	}

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

	llvm::WriteBitcodeToFile(*llvmModule, os);
	return 0;
}
