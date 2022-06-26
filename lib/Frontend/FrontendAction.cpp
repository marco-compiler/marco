#include "marco/Frontend/FrontendAction.h"
#include "marco/Diagnostic/Printer.h"
#include "marco/AST/Passes.h"
#include "marco/Parser/Parser.h"
#include "marco/Codegen/Bridge.h"
#include "marco/Codegen/Conversion/Passes.h"
#include "marco/Codegen/Transforms/Passes.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Frontend/CompilerInstance.h"
#include "marco/Frontend/FrontendActions.h"
#include "marco/Frontend/FrontendOptions.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Transforms/Utils.h"

static bool exec(const char* cmd, std::string& result)
{
  std::array<char, 128> buffer;
  FILE* pipe = popen(cmd, "r");

  if (!pipe) {
    return false;
  }

  while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    result += buffer.data();
  }

  return pclose(pipe)==0;
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

    auto result = exec(cmd.c_str(), ci.getFlattened());

    if (!result) {
      unsigned int diagID = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Fatal,
          "OMC flattening failed");

      ci.getDiagnostics().Report(diagID);
      llvm::errs() << ci.getFlattened();
    }

    return result;
  }

  bool FrontendAction::runParse()
  {
    CompilerInstance& ci = instance();

    diagnostic::DiagnosticEngine diagnostics(std::make_unique<diagnostic::Printer>());
    auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(ci.getFlattened()));
    parser::Parser parser(diagnostics, sourceFile);
    auto cls = parser.parseRoot();

    if (!cls.hasValue()) {
      unsigned int diagID = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Fatal,
          "AST generation failed");

      ci.getDiagnostics().Report(diagID);
      return false;
    }

    instance().setAST(std::move(*cls));
    return true;
  }

  bool FrontendAction::runFrontendPasses()
  {
    CompilerInstance& ci = instance();
    diagnostic::DiagnosticEngine diagnostics(std::make_unique<diagnostic::Printer>());

    marco::ast::PassManager frontendPassManager;
    frontendPassManager.addPass(ast::createTypeInferencePass(diagnostics));
    frontendPassManager.addPass(ast::createTypeCheckingPass(diagnostics));
    frontendPassManager.addPass(ast::createSemanticAnalysisPass(diagnostics));
    frontendPassManager.addPass(ast::createConstantFoldingPass(diagnostics));

    if (!frontendPassManager.run(instance().getAST())) {
      unsigned int diagID = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Fatal,
          "Frontend passes failed");

      ci.getDiagnostics().Report(diagID);
      return false;
    }

    return true;
  }

  bool FrontendAction::runASTConversion()
  {
    CompilerInstance& ci = instance();

    codegen::CodegenOptions options;

    marco::codegen::lowering::Bridge bridge(ci.getMLIRContext(), options);
    bridge.lower(*ci.getAST());
    instance().setMLIRModule(std::move(bridge.getMLIRModule()));

    return true;
  }

  bool FrontendAction::runDialectConversion()
  {
    CompilerInstance& ci = instance();

    auto& codegenOptions = ci.getCodegenOptions();
    auto& simulationOptions = ci.getSimulationOptions();

    mlir::PassManager passManager(&ci.getMLIRContext());

    passManager.addPass(codegen::createAutomaticDifferentiationPass());

    // Model solving
    codegen::ModelSolvingOptions modelSolvingOptions;

    modelSolvingOptions.startTime = simulationOptions.startTime;
    modelSolvingOptions.endTime = simulationOptions.endTime;
    modelSolvingOptions.timeStep = simulationOptions.timeStep;

    modelSolvingOptions.emitMain = codegenOptions.generateMain;
    modelSolvingOptions.variableFilter = &ci.getFrontendOptions().variableFilter;

    modelSolvingOptions.solver = simulationOptions.solver;

    modelSolvingOptions.ida.equidistantTimeGrid = simulationOptions.ida.equidistantTimeGrid;

    passManager.addPass(codegen::createModelSolvingPass(modelSolvingOptions));

    // Functions scalarization pass
    codegen::FunctionScalarizationOptions functionScalarizationOptions;
    functionScalarizationOptions.assertions = codegenOptions.assertions;
    passManager.addPass(codegen::createFunctionScalarizationPass(functionScalarizationOptions));

    // Insert explicit casts where needed
    passManager.addPass(codegen::createExplicitCastInsertionPass());

    if (codegenOptions.inlining) {
      // Inline the functions with the 'inline' annotation

      // The inlining pass is incompatible with the new Modelica dialect, as functions
      // input members are now declared within the body and not in the entry region
      // signature. Moreover, input members have now "member" type, thus not matching
      // arguments of the function call. A new intermediate (internal) dialect between
      // Modelica and std is needed, in order to define a std-like function with
      // additional attributes (such as the inlining one).
      //passManager.addPass(mlir::createInlinerPass());
    }

    passManager.addPass(mlir::createCanonicalizerPass());

    if (codegenOptions.cse) {
      passManager.addNestedPass<mlir::modelica::FunctionOp>(mlir::createCSEPass());
      passManager.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
    }

    // Place the deallocation instructions for the arrays
    passManager.addPass(codegen::createArrayDeallocationPass());

    // Modelica to CF conversion
    codegen::ModelicaToCFOptions modelicaToCFOptions;
    modelicaToCFOptions.bitWidth = codegenOptions.bitWidth;
    modelicaToCFOptions.outputArraysPromotion = codegenOptions.outputArraysPromotion;
    modelicaToCFOptions.inlining = codegenOptions.inlining;

    passManager.addPass(codegen::createModelicaToCFPass(modelicaToCFOptions));

    // Modelica To LLVM conversion
    codegen::ModelicaToLLVMOptions modelicaToLLVMOptions;
    modelicaToLLVMOptions.assertions = codegenOptions.assertions;

    passManager.addPass(codegen::createModelicaToLLVMPass(modelicaToLLVMOptions));
    passManager.addPass(codegen::createIDAToLLVMPass());

    if (codegenOptions.omp) {
      // Use OpenMP for parallel loops
      passManager.addNestedPass<mlir::func::FuncOp>(mlir::createConvertSCFToOpenMPPass());
    }

    passManager.addPass(mlir::arith::createConvertArithmeticToLLVMPass());
    passManager.addPass(mlir::createConvertSCFToCFPass());
    passManager.addPass(mlir::createConvertFuncToLLVMPass());
    passManager.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (!codegenOptions.debug) {
      // Remove the debug information if a non-debuggable executable has been requested
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
