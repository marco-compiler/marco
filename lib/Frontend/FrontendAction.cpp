#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Utils.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "marco/AST/Parser.h"
#include "marco/AST/Passes.h"
#include "marco/Frontend/CompilerInstance.h"
#include "marco/Frontend/FrontendAction.h"
#include "marco/Frontend/FrontendActions.h"
#include "marco/Frontend/FrontendOptions.h"
#include "marco/Codegen/Bridge.h"
#include "marco/Codegen/Conversion/Passes.h"
#include "marco/Codegen/dialects/modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Path.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

bool exec(const char* cmd, std::string& result)
{
  std::array<char, 128> buffer;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);

  if (!pipe) {
    return false;
  }

  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }

  return true;
}

namespace marco::frontend
{
  bool FrontendAction::beginAction()
  {
    return true;
  }

  bool FrontendAction::runFlattening()
  {
    CompilerInstance& ci = instance();

    if (ci.getFrontendOptions().omcBypass) {
      const auto& inputs = ci.getFrontendOptions().inputs;

      if (inputs.size() > 1) {
        unsigned int diagID = ci.getDiagnostics().getCustomDiagID(
            clang::DiagnosticsEngine::Fatal,
            "MARCO can receive only one input flattened file");

        ci.getDiagnostics().Report(diagID);
        return false;
      }

      auto errorOrBuffer = llvm::MemoryBuffer::getFileOrSTDIN(inputs[0].file());
      auto buffer = llvm::errorOrToExpected(std::move(errorOrBuffer));

      if (!buffer) {
        unsigned int diagID = ci.getDiagnostics().getCustomDiagID(
            clang::DiagnosticsEngine::Fatal,
            "Can't open the input file");

        ci.getDiagnostics().Report(diagID);
        llvm::consumeError(buffer.takeError());
        return false;
      }

      ci.setFlattened((*buffer)->getBuffer().str());
      return true;
    }

    llvm::SmallString<256> cmd;

    if (auto path = ci.getFrontendOptions().omcPath; !path.empty()) {
      llvm::sys::path::append(cmd, path, "omc");
    } else {
      llvm::sys::path::append(cmd, "omc");
    }

    for (const auto& input : ci.getFrontendOptions().inputs) {
      cmd += " \"" + input.file().str() + "\"";
    }

    if (const auto& modelName = ci.getSimulationOptions().modelName; modelName.empty()) {
      unsigned int diagID = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Warning,
          "Model name not specified");

      ci.getDiagnostics().Report(diagID);
    } else {
      cmd += " +i=" + ci.getSimulationOptions().modelName;
    }

    if (const auto& args = ci.getFrontendOptions().omcCustomArgs; args.empty()) {
      cmd += " -f";
      cmd += " -d=nonfScalarize,arrayConnect,combineSubscripts,printRecordTypes";
      cmd += " --newBackend";
      cmd += " --showStructuralAnnotations";
    } else {
      for (const auto& arg : args) {
        cmd += " " + arg;
      }
    }

    return exec(cmd.c_str(), ci.getFlattened());
  }

  bool FrontendAction::runParse()
  {
    CompilerInstance& ci = instance();

    ast::Parser parser(ci.getFlattened());
    auto cls = parser.classDefinition();

    if (!cls) {
      unsigned int diagID = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Fatal,
          "AST generation failed");

      ci.getDiagnostics().Report(diagID);
      llvm::consumeError(cls.takeError());
      return false;
    }

    instance().setAST(std::move(*cls));
    return true;
  }

  bool FrontendAction::runFrontendPasses()
  {
    CompilerInstance& ci = instance();

    marco::ast::PassManager frontendPassManager;
    frontendPassManager.addPass(ast::createTypeCheckingPass());
    frontendPassManager.addPass(ast::createConstantFolderPass());
    auto error = frontendPassManager.run(instance().getAST());

    if (error) {
      unsigned int diagID = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Fatal,
          "Frontend passes failed");

      ci.getDiagnostics().Report(diagID);
      llvm::consumeError(std::move(error));
      return false;
    }

    return true;
  }

  bool FrontendAction::runASTConversion()
  {
    CompilerInstance& ci = instance();

    codegen::CodegenOptions options;
    options.startTime = ci.getSimulationOptions().startTime;
    options.endTime = ci.getSimulationOptions().endTime;
    options.timeStep = ci.getSimulationOptions().timeStep;

    marco::codegen::LoweringBridge bridge(ci.getMLIRContext(), options);
    auto module = bridge.run(ci.getAST());

    if (!module) {
      unsigned int diagID = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Fatal,
          "MLIR conversion failed");

      ci.getDiagnostics().Report(diagID);
      return false;
    }

    instance().setMLIRModule(std::make_unique<mlir::ModuleOp>(std::move(*module)));
    return true;
  }

  bool FrontendAction::runDialectConversion()
  {
    CompilerInstance& ci = instance();

    auto& codegenOptions = ci.getCodegenOptions();
    mlir::PassManager passManager(&ci.getMLIRContext());

    passManager.addPass(codegen::createAutomaticDifferentiationPass());

    // Solve model pass
    codegen::SolveModelOptions solveModelOptions;
    solveModelOptions.emitMain = ci.getCodegenOptions().generateMain;
    solveModelOptions.variableFilter = &ci.getFrontendOptions().variableFilter;
    passManager.addNestedPass<codegen::modelica::ModelOp>(codegen::createSolveModelPass(solveModelOptions));

    // Functions vectorization pass
    codegen::FunctionsVectorizationOptions functionsVectorizationOptions;
    functionsVectorizationOptions.assertions = ci.getCodegenOptions().assertions;
    passManager.addPass(codegen::createFunctionsVectorizationPass(functionsVectorizationOptions));

    passManager.addPass(codegen::createExplicitCastInsertionPass());

    if (codegenOptions.outputArraysPromotion) {
      passManager.addPass(codegen::createOutputArraysPromotionPass());
    }

    if (codegenOptions.inlining) {
      passManager.addPass(mlir::createInlinerPass());
    }

    passManager.addPass(mlir::createCanonicalizerPass());

    if (codegenOptions.cse) {
      passManager.addNestedPass<codegen::modelica::FunctionOp>(mlir::createCSEPass());
    }

    passManager.addPass(codegen::createFunctionConversionPass());

    // The buffer deallocation pass must be placed after the Modelica's
    // functions and members conversion, so that we can operate on an IR
    // without hidden allocs and frees.
    // However the pass must also be placed before the conversion of the
    // more common Modelica operations (i.e. add, sub, call, etc.), in
    // order to take into consideration their memory effects.
    passManager.addPass(codegen::createBufferDeallocationPass());

    // Modelica conversion pass
    codegen::ModelicaConversionOptions modelicaConversionOptions;
    modelicaConversionOptions.assertions = ci.getCodegenOptions().assertions;
    passManager.addPass(codegen::createModelicaConversionPass(modelicaConversionOptions));

    if (codegenOptions.omp) {
      passManager.addNestedPass<mlir::FuncOp>(mlir::createConvertSCFToOpenMPPass());
    }

    passManager.addPass(codegen::createLowerToCFGPass());
    passManager.addNestedPass<mlir::FuncOp>(mlir::createConvertMathToLLVMPass());

    // Conversion to LLVM dialect
    codegen::ModelicaToLLVMConversionOptions llvmLoweringOptions;
    llvmLoweringOptions.assertions = ci.getCodegenOptions().assertions;
    llvmLoweringOptions.emitCWrappers = ci.getCodegenOptions().cWrappers;
    passManager.addPass(codegen::createLLVMLoweringPass(llvmLoweringOptions));

    if (!codegenOptions.debug) {
      passManager.addPass(mlir::createStripDebugInfoPass());
    }

    if (auto status = passManager.run(ci.getMLIRModule()); mlir::failed(status)) {
      unsigned int diagID = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Fatal,
          "Modelica dialect conversion failure");

      ci.getDiagnostics().Report(diagID);
      return false;
    }

    return true;
  }

  bool FrontendAction::runLLVMIRGeneration()
  {
    CompilerInstance& ci = instance();

    // Register the conversions to LLVM IR
    mlir::registerLLVMDialectTranslation(ci.getMLIRContext());
    mlir::registerOpenMPDialectTranslation(ci.getMLIRContext());

    // Initialize LLVM targets
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Convert to LLVM IR
    auto llvmModule = mlir::translateModuleToLLVMIR(ci.getMLIRModule(), ci.getLLVMContext());

    if (!llvmModule) {
      unsigned int diagId = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "Failed to emit LLVM IR");

      ci.getDiagnostics().Report(diagId);
      return false;
    }

    // Optimize the IR
    auto optLevel = ci.getCodegenOptions().optLevel;
    auto optPipeline = mlir::makeOptimizingTransformer(optLevel.time, optLevel.size, nullptr);

    if (auto error = optPipeline(llvmModule.get())) {
      unsigned int diagId = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "Failed to optimize LLVM IR");

      ci.getDiagnostics().Report(diagId);
      llvm::consumeError(std::move(error));
      return false;
    }

    instance().setLLVMModule(std::move(llvmModule));
    return true;
  }
}
