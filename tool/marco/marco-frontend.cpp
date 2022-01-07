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
#include <marco/frontend/Options.h>
#include <marco/frontend/TextDiagnosticPrinter.h>

using namespace llvm;
using namespace marco;
using namespace marco::frontend;
using namespace std;

//static cl::OptionCategory driverModeOptions("Driver mode");

/*
static cl::opt<codegen::Solver> solver(
    cl::desc("Solvers:"),
    cl::values(
        clEnumValN(codegen::ForwardEuler, "forward-euler", "Forward Euler (default)"),
        clEnumValN(codegen::CleverDAE, "clever-dae", "Clever DAE")),
    cl::init(codegen::ForwardEuler),
    cl::cat(modelSolvingOptions));

static cl::opt<string>
    filter("variableFilter", cl::desc("Variable filtering expression"), cl::init(""), cl::cat(modelSolvingOptions));

static cl::OptionCategory codeGenOptions("Code generation options");

static cl::opt<bool> emitMain(
    "emit-main",
    cl::desc("Whether to emit the main function that will start the simulation (default: true)"),
    cl::init(true),
    cl::cat(codeGenOptions));

static cl::opt<bool>
    x86("32", cl::desc("Use 32-bit values instead of 64-bit ones"), cl::init(false), cl::cat(codeGenOptions));

static cl::opt<bool>
    inlining("no-inlining", cl::desc("Disable the inlining pass"), cl::init(false), cl::cat(codeGenOptions));

static cl::opt<bool> resultBuffersToArgs(
    "no-result-buffers-to-args",
    cl::desc("Don't move the static output buffer to input arguments"),
    cl::init(false),
    cl::cat(codeGenOptions));

static cl::opt<bool> cse("no-cse", cl::desc("Disable CSE pass"), cl::init(false), cl::cat(codeGenOptions));

static cl::opt<bool> openmp("omp", cl::desc("Enable OpenMP usage"), cl::init(false), cl::cat(codeGenOptions));

static cl::opt<bool> disableRuntimeLibrary(
    "disable-runtime-library",
    cl::desc(
        "Avoid the calls to the external runtime library functions (only when a native implementation of the operation exists)"),
    cl::init(false),
    cl::cat(codeGenOptions));

static cl::opt<bool>
    emitCWrappers("emit-c-wrappers", cl::desc("Emit C wrappers"), cl::init(false), cl::cat(codeGenOptions));

enum OptLevel
{
  O0, O1, O2, O3
};

static cl::opt<OptLevel> optimizationLevel(
    cl::desc("Optimization level:"),
    cl::values(
        clEnumValN(O0, "O0", "No optimizations"),
        clEnumValN(O1, "O1", "Trivial optimizations"),
        clEnumValN(O2, "O2", "Default optimizations"),
        clEnumValN(O3, "O3", "Expensive optimizations")),
    cl::cat(codeGenOptions),
    cl::init(O2));

static cl::OptionCategory debugOptions("Debug options");

static cl::opt<bool> printParsedAST
    ("print-parsed-ast", cl::desc("Print the AST right after being parsed"), cl::init(false), cl::cat(debugOptions));
static cl::opt<bool> printLegalizedAST(
    "print-legalized-ast",
    cl::desc("Print the AST after it has been legalized"),
    cl::init(false),
    cl::cat(debugOptions));
static cl::opt<bool> printModelicaDialectIR(
    "print-modelica",
    cl::desc("Print the Modelica dialect IR obtained right after the AST lowering"),
    cl::init(false),
    cl::cat(debugOptions));
static cl::opt<bool>
    printLLVMDialectIR("print-llvm", cl::desc("Print the LLVM dialect IR"), cl::init(false), cl::cat(debugOptions));

static cl::opt<bool>
    debug("d", cl::desc("Keep debug information in the final IR"), cl::init(false), cl::cat(debugOptions));

static cl::opt<bool> enableAssertions(
    "enable-assertions",
    cl::desc("Enable assertions (default: true for O0, false otherwise)"),
    cl::init(true),
    cl::cat(debugOptions));

static cl::OptionCategory simulationOptions("Simulation options");

static cl::opt<double>
    startTime("start-time", cl::desc("Start time (in seconds) (default: 0)"), cl::init(0), cl::cat(simulationOptions));
static cl::opt<double>
    endTime("end-time", cl::desc("End time (in seconds) (default: 10)"), cl::init(10), cl::cat(simulationOptions));
static cl::opt<double>
    timeStep("time-step", cl::desc("Time step (in seconds) (default: 0.1)"), cl::init(0.1), cl::cat(simulationOptions));

    */

static llvm::ExitOnError exitOnErr;

struct ModelicaLoweringOptions
{
  bool x64 = true;
  codegen::SolveModelOptions solveModelOptions = codegen::SolveModelOptions::getDefaultOptions();
  codegen::FunctionsVectorizationOptions
      functionsVectorizationOptions = codegen::FunctionsVectorizationOptions::getDefaultOptions();
  bool inlining = true;
  bool resultBuffersToArgs = true;
  bool cse = true;
  bool openmp = false;
  codegen::ModelicaConversionOptions conversionOptions = codegen::ModelicaConversionOptions::getDefaultOptions();
  codegen::ModelicaToLLVMConversionOptions llvmOptions = codegen::ModelicaToLLVMConversionOptions::getDefaultOptions();
  bool debug = true;

  unsigned int getBitWidth() const
  {
    if (x64) {
      return 64;
    }

    return 32;
  }

  static const ModelicaLoweringOptions& getDefaultOptions()
  {
    static ModelicaLoweringOptions options;
    return options;
  }
};

static mlir::LogicalResult lower(mlir::ModuleOp& module, ModelicaLoweringOptions options);

struct Options : public FrontendOptions {

  /*
  llvm::StringRef outputFile() const override
  {
    return ::outputFile;
  }
   */
};

extern int mc1_main(llvm::ArrayRef<const char*> argv, const char* argv0)
{
  // Create CompilerInstance
  std::unique_ptr<CompilerInstance> instance(new CompilerInstance());

  // Create diagnostics engine for the frontend driver
  instance->CreateDiagnostics();

  // We will buffer diagnostics from argument parsing so that we can output
  // them using a well-formed diagnostic object.
  TextDiagnosticBuffer* diagnosticBuffer = new TextDiagnosticBuffer();

  // Create CompilerInvocation - use a dedicated instance of DiagnosticsEngine
  // for parsing the arguments
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagnosticID(new clang::DiagnosticIDs());
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagnosticOptions = new clang::DiagnosticOptions();
  clang::DiagnosticsEngine diagnosticEngine(diagnosticID, &*diagnosticOptions, diagnosticBuffer);

  bool success = CompilerInvocation::CreateFromArgs(instance->invocation(), argv, diagnosticEngine);

  diagnosticBuffer->FlushDiagnostics(instance->diagnostics());

  if (!success) {
    return 1;
  }

  // Execute the frontend actions
  success = ExecuteCompilerInvocation(instance.get());

  // Delete output files to free Compiler Instance
  instance->ClearOutputFiles(false);

  return !success;
}

static std::string GetExecutablePath(const char* argv0)
{
  // This just needs to be some symbol in the binary
  void* p = (void*) (intptr_t) GetExecutablePath;
  return llvm::sys::fs::getMainExecutable(argv0, p);
}

mlir::LogicalResult lower(mlir::ModuleOp& module, ModelicaLoweringOptions options)
{
  mlir::PassManager passManager(module.getContext());

  //passManager.addPass(codegen::createAutomaticDifferentiationPass());
  passManager.addNestedPass<codegen::modelica::ModelOp>(codegen::createSolveModelPass(options.solveModelOptions));

  /*
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
    */

  return passManager.run(module);
}
