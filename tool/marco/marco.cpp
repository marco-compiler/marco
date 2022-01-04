#include <iostream>
#include <llvm/ADT/SmallVector.h>
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
#include <marco/ast/Parser.h>
#include <marco/ast/Passes.h>
#include <marco/codegen/CodeGen.h>
#include <marco/codegen/Passes.h>
#include <marco/utils/VariableFilter.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

#include "clang/Driver/Driver.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <marco/frontend/CompilerInstance.h>
#include <marco/frontend/CompilerInvocation.h>
#include <marco/frontend/TextDiagnosticBuffer.h>
#include <marco/frontendTool/Utils.h>
#include <clang/Driver/DriverDiagnostic.h>
#include <llvm/Option/Arg.h>
#include <llvm/Option/ArgList.h>
#include <llvm/Option/OptTable.h>

#include <marco/frontend/CompilerInvocation.h>
#include <marco/frontend/TextDiagnosticPrinter.h>

using namespace llvm;
using namespace marco;
using namespace marco::frontend;
using namespace std;

bool isFrontendTool(llvm::StringRef tool)
{
  return tool == "-mc1";
}

extern bool isFrontendOption(llvm::StringRef option);
extern int marcoFrontend(llvm::ArrayRef<const char*> argv, const char* argv0);

static int executeMarcoFrontend(llvm::StringRef tool, int argc, const char** argv)
{
  if (tool == "-mc1") {
    return marcoFrontend(makeArrayRef(argv).slice(2), argv[0]);
  }

  // Reject unknown tools.
  // At the moment it only supports mc1. Any mc1[*] is rejected.

  llvm::errs() << "error: unknown integrated tool '" << tool << "'. "
               << "Valid tools include '-mc1'.\n";
  return 1;
}

static std::string GetExecutablePath(const char* argv0)
{
  // This just needs to be some symbol in the binary
  void* p = (void*) (intptr_t) GetExecutablePath;
  return llvm::sys::fs::getMainExecutable(argv0, p);
}

// This lets us create the DiagnosticsEngine with a properly-filled-out
// DiagnosticOptions instance
static clang::DiagnosticOptions *CreateAndPopulateDiagOpts(
    llvm::ArrayRef<const char *> argv) {
  auto *diagOpts = new clang::DiagnosticOptions;

  // Ignore missingArgCount and the return value of ParseDiagnosticArgs.
  // Any errors that would be diagnosed here will also be diagnosed later,
  // when the DiagnosticsEngine actually exists.
  unsigned missingArgIndex, missingArgCount;

  llvm::opt::InputArgList args = clang::driver::getDriverOptTable().ParseArgs(
      argv.slice(1), missingArgIndex, missingArgCount);

  return diagOpts;
}



int main(int argc, const char** argv)
{
  // Initialize variables to call the driver
  llvm::InitLLVM x(argc, argv);

  llvm::ArrayRef args(argv, argv + argc);

  // Check if MARCO is in the frontend mode
  auto firstArg = std::find_if(args.begin() + 1, args.end(), [](const char* arg) {
    return arg != nullptr;
  });

  if (firstArg != args.end()) {
    if (llvm::StringRef(*firstArg).startswith("-mc1")) {
      return executeMarcoFrontend(llvm::StringRef(*firstArg), argc, argv);
    }
  }

  // Not in the frontend mode. Continue in the compiler driver mode.

  // Create the diagnostics engine for the driver
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOpts = CreateAndPopulateDiagOpts(args);
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagID(new clang::DiagnosticIDs());

  marco::frontend::TextDiagnosticPrinter* diagClient =
      new marco::frontend::TextDiagnosticPrinter(llvm::errs(), &*diagOpts);

  diagClient->set_prefix(std::string(llvm::sys::path::stem(GetExecutablePath(args[0]))));

  clang::DiagnosticsEngine diags(diagID, &*diagOpts, diagClient);

  // Prepare the driver
  clang::driver::ParsedClangName targetAndMode("marco", "--driver-mode=marco");
  std::string driverPath = GetExecutablePath(args[0]);
  clang::driver::Driver theDriver(driverPath, llvm::sys::getDefaultTargetTriple(), diags, "MARCO");
  theDriver.setTargetAndMode(targetAndMode);
  std::unique_ptr<clang::driver::Compilation> c(theDriver.BuildCompilation(args));
  llvm::SmallVector<std::pair<int, const clang::driver::Command*>, 4> failingCommands;

  // Run the driver
  int res = 1;
  bool isCrash = false;
  res = theDriver.ExecuteCompilation(*c, failingCommands);

  for (const auto& p: failingCommands) {
    int CommandRes = p.first;
    const clang::driver::Command* failingCommand = p.second;
    if (!res) {
      res = CommandRes;
    }

    // If result status is < 0 (e.g. when sys::ExecuteAndWait returns -1),
    // then the driver command signalled an error. On Windows, abort will
    // return an exit code of 3. In these cases, generate additional diagnostic
    // information if possible.
    isCrash = CommandRes < 0;
    #ifdef _WIN32
    isCrash |= CommandRes == 3;
    #endif
    if (isCrash) {
      theDriver.generateCompilationDiagnostics(*c, *failingCommand);
      break;
    }
  }

  diags.getClient()->finish();

  // If we have multiple failing commands, we return the result of the first
  // failing command.
  return res;





  /*
  // Setup the command line options
	llvm::SmallVector<const cl::OptionCategory*> categories;
	categories.push_back(&modelSolvingOptions);
	categories.push_back(&codeGenOptions);
	categories.push_back(&debugOptions);
	categories.push_back(&simulationOptions);
	HideUnrelatedOptions(categories);

	cl::ParseCommandLineOptions(argc, argv);

	error_code error;
	llvm::raw_fd_ostream os(outputFile, error, llvm::sys::fs::OF_None);

	if (error)
	{
		llvm::errs() << error.message();
		return -1;
	}

	auto variableFilter = exitOnErr(VariableFilter::fromString(filter));

  // Parse the input files
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
	ModelicaLoweringOptions loweringOptions;
	loweringOptions.solveModelOptions.emitMain = emitMain;
	loweringOptions.solveModelOptions.variableFilter = &variableFilter;
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

	if (mlir::failed(lower(*module, loweringOptions)))
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
   */
}

/*
mlir::LogicalResult lower(mlir::ModuleOp& module, ModelicaLoweringOptions options)
{
  mlir::PassManager passManager(module.getContext());

  //passManager.addPass(codegen::createAutomaticDifferentiationPass());
  passManager.addNestedPass<codegen::modelica::ModelOp>(codegen::createSolveModelPass(options.solveModelOptions));

  passManager.addPass(codegen::createFunctionsVectorizationPass(options.functionsVectorizationOptions));
  passManager.addPass(codegen::createExplicitCastInsertionPass());

  if (options.resultBuffersToArgs)
    passManager.addPass(codegen::createResultBuffersToArgsPass());

  if (options.inlining)
    passManager.addPass(mlir::createInlinerPass());

  passManager.addPass(mlir::createCanonicalizerPass());

  if (options.cse)
    passManager.addNestedPass<codegen::modelica::FunctionOp>(mlir::createCSEPass());

  passManager.addPass(codegen::createFunctionConversionPass());

  // The buffer deallocation pass must be placed after the Modelica's
  // functions and members conversion, so that we can operate on an IR
  // without hidden allocs and frees.
  // However the pass must also be placed before the conversion of the
  // more common Modelica operations (i.e. add, sub, call, etc.), in
  // order to take into consideration their memory effects.
  passManager.addPass(codegen::createBufferDeallocationPass());

  passManager.addPass(codegen::createModelicaConversionPass(options.conversionOptions, options.getBitWidth()));

  if (options.openmp)
    passManager.addNestedPass<mlir::FuncOp>(mlir::createConvertSCFToOpenMPPass());

  passManager.addPass(codegen::createLowerToCFGPass(options.getBitWidth()));
  passManager.addNestedPass<mlir::FuncOp>(mlir::createConvertMathToLLVMPass());
  passManager.addPass(codegen::createLLVMLoweringPass(options.llvmOptions, options.getBitWidth()));

  if (!options.debug)
    passManager.addPass(mlir::createStripDebugInfoPass());

  return passManager.run(module);
}
    */

/*
static int execute(std::string cmd, std::string& output);
static int runOMC();
static int runMARCO();
static int runBackend();

int mainNew(int argc, const char** argv)
{
  llvm::InitLLVM x(argc, argv);

  llvm::SmallVector<const cl::OptionCategory*> categories;
  categories.push_back(&openModelicaOptions);
  categories.push_back(&modelSolvingOptions);
  categories.push_back(&codeGenOptions);
  categories.push_back(&debugOptions);
  categories.push_back(&simulationOptions);
  HideUnrelatedOptions(categories);

  cl::ParseCommandLineOptions(argc, argv, "MARCO - MLIR-based Modelica compiler");

  switch (driverMode) {
    case OMC:
      return runOMC();

    case MARCO:
      return runMARCO();

    case BACKEND:
      return runBackend();
  }

  std::cerr << "Unknown driver mode" << std::endl;
  return EXIT_FAILURE;
}

int execute(std::string cmd, std::string& output) {
  const int bufsize=128;
  std::array<char, bufsize> buffer;

  auto pipe = popen(cmd.c_str(), "r");

  if (!pipe) {
    return EXIT_FAILURE;
  }

  size_t count;

  do {
    if ((count = fread(buffer.data(), 1, bufsize, pipe)) > 0) {
      output.insert(output.end(), std::begin(buffer), std::next(std::begin(buffer), count));
    }
  } while(count > 0);

  return pclose(pipe) == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int runOMC()
{
  return EXIT_SUCCESS;
}

int runMARCO()
{
  return EXIT_SUCCESS;
}

int runBackend()
{
  return EXIT_SUCCESS;
}
*/