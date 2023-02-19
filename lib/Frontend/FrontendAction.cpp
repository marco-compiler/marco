#include "marco/Frontend/FrontendAction.h"
#include "marco/Diagnostic/Printer.h"
#include "marco/AST/Passes.h"
#include "marco/Parser/Parser.h"
#include "marco/Codegen/Bridge.h"
#include "marco/Codegen/Conversion/Passes.h"
#include "marco/Codegen/Transforms/Passes.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/Simulation/SimulationDialect.h"
#include "marco/Frontend/CompilerInstance.h"
#include "marco/Frontend/FrontendActions.h"
#include "marco/Frontend/FrontendOptions.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Transforms/Utils.h"

using namespace ::marco;
using namespace ::marco::diagnostic;
using namespace ::marco::frontend;

//===----------------------------------------------------------------------===//
// Messages
//===----------------------------------------------------------------------===//

namespace
{
  class CantOpenInputFileMessage : public Message
  {
    public:
      CantOpenInputFileMessage(llvm::StringRef file)
          : file(file.str())
      {
      }

      void print(PrinterInstance* printer) const override
      {
        auto& os = printer->getOutputStream();
        os << "Unable to open input file '" << file << "'" << "\n";
      }

    private:
      std::string file;
  };

  class FlatteningFailureMessage : public Message
  {
    public:
      FlatteningFailureMessage(llvm::StringRef error)
          : error(error.str())
      {
      }

      void print(PrinterInstance* printer) const override
      {
        auto& os = printer->getOutputStream();
        os << "OMC flattening failed";

        if (!error.empty()) {
          os << "\n\n" << error << "\n";
        }
      }

    private:
      std::string error;
  };
}

//===----------------------------------------------------------------------===//
// FrontendAction
//===----------------------------------------------------------------------===//

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
        ci.getDiagnostics().emitFatalError<GenericStringMessage>(
            "MARCO can receive only one input flattened file");

        return false;
      }

      auto errorOrBuffer = llvm::MemoryBuffer::getFileOrSTDIN(inputs[0].file());
      auto buffer = llvm::errorOrToExpected(std::move(errorOrBuffer));

      if (!buffer) {
        ci.getDiagnostics().emitFatalError<CantOpenInputFileMessage>(inputs[0].file());
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
      ci.getDiagnostics().emitWarning<GenericStringMessage>("Model name not specified");
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
      ci.getDiagnostics().emitFatalError<FlatteningFailureMessage>(ci.getFlattened());
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

    if (!cls.has_value()) {
      ci.getDiagnostics().emitFatalError<GenericStringMessage>("AST generation failed");
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
    frontendPassManager.addPass(ast::createRaggedFlatteningPass(diagnostics));
    frontendPassManager.addPass(ast::createTypeInferencePass(diagnostics));
    frontendPassManager.addPass(ast::createTypeCheckingPass(diagnostics));
    frontendPassManager.addPass(ast::createSemanticAnalysisPass(diagnostics));
    frontendPassManager.addPass(ast::createInliningPass(diagnostics));
    frontendPassManager.addPass(ast::createConstantFoldingPass(diagnostics));

    if (!frontendPassManager.run(instance().getAST())) {
      ci.getDiagnostics().emitFatalError<GenericStringMessage>("Frontend passes failed");
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

    // Set the target triple inside the MLIR module.
    instance().getMLIRModule()->setAttr(
        mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
        mlir::StringAttr::get(
            &instance().getMLIRContext(),
            ci.getCodegenOptions().target));

    // Set the data layout inside the MLIR module.
    instance().getMLIRModule()->setAttr(
        mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
        mlir::StringAttr::get(
            &instance().getMLIRContext(),
            getDataLayoutString()));

    // Create the pass manager and populate it with the appropriate
    // transformations.
    mlir::PassManager passManager(&ci.getMLIRContext());

    if (!ci.getCodegenOptions().debug) {
      // Remove the debug information if a non-debuggable executable has been
      // requested. By doing this at the beginning of the compilation pipeline
      // we reduce the time needed for the pass itself, as the code inevitably
      // grows while traversing the pipeline.
      passManager.addPass(mlir::createStripDebugInfoPass());
    }

    // Try to simplify the starting IR.
    passManager.addPass(mlir::createCanonicalizerPass());

    if (ci.getCodegenOptions().readOnlyVariablesPropagation) {
      // Propagate the read-only variables and try to fold the constants right
      // after.
      passManager.addPass(createReadOnlyVariablesPropagationPass());
      passManager.addPass(mlir::createCanonicalizerPass());
    }

    passManager.addPass(createAutomaticDifferentiationPass());
    passManager.addPass(createModelLegalizationPass());
    passManager.addPass(createMatchingPass());
    passManager.addPass(createVariablesPromotionPass());
    passManager.addPass(createCyclesSolvingPass());
    passManager.addPass(createSchedulingPass());

    // Apply the selected solver.
    passManager.addPass(
        llvm::StringSwitch<std::unique_ptr<mlir::Pass>>(
            ci.getSimulationOptions().solver)
            .Case("euler-forward", createEulerForwardPass())
            .Case("ida", createIDAPass())
            .Default(createEulerForwardPass()));

    passManager.addPass(createFunctionScalarizationPass());
    passManager.addPass(createExplicitCastInsertionPass());
    passManager.addPass(mlir::createCanonicalizerPass());

    if (ci.getCodegenOptions().cse) {
      passManager.addNestedPass<mlir::modelica::FunctionOp>(
          mlir::createCSEPass());

      passManager.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
    }

    passManager.addPass(mlir::modelica::createArrayDeallocationPass());
    passManager.addPass(createModelicaToCFConversionPass());

    if (ci.getCodegenOptions().inlining) {
      // Inline the functions with the 'inline' annotation.
      passManager.addPass(mlir::createInlinerPass());
    }

    passManager.addPass(createModelicaToVectorConversionPass());
    passManager.addPass(createModelicaToArithConversionPass());
    passManager.addPass(createModelicaToFuncConversionPass());
    passManager.addPass(createModelicaToMemRefConversionPass());

    passManager.addPass(createIDAToFuncConversionPass());

    if (ci.getCodegenOptions().omp) {
      // Use OpenMP for parallel loops.
      passManager.addNestedPass<mlir::func::FuncOp>(
          mlir::createConvertSCFToOpenMPPass());
    }

    passManager.addNestedPass<mlir::func::FuncOp>(
        createVectorToSCFConversionPass());

    passManager.addPass(createVectorToLLVMConversionPass());

    passManager.addPass(createArithToLLVMConversionPass());
    passManager.addPass(createMemRefToLLVMConversionPass());
    passManager.addPass(mlir::createConvertSCFToCFPass());

    //passManager.addPass(createFuncToLLVMConversionPass(true));
    passManager.addPass(createFuncToLLVMConversionPass(false));

    passManager.addPass(mlir::cf::createConvertControlFlowToLLVMPass());

    // Convert the MARCO dialects to LLVM dialect.
    passManager.addPass(createModelicaToLLVMConversionPass());
    passManager.addPass(createIDAToLLVMConversionPass());
    passManager.addPass(createKINSOLToLLVMConversionPass());

    // Now that the Simulation dialect doesn't have dependencies from Modelica
    // or the solvers, we can proceed converting it.
    passManager.addPass(createSimulationToFuncConversionPass());

    // Convert the non-LLVM operations that may have been introduced by the
    // last conversions.
    passManager.addNestedPass<mlir::func::FuncOp>(
        createArithToLLVMConversionPass());

    passManager.addPass(createFuncToLLVMConversionPass(false));
    passManager.addPass(mlir::cf::createConvertControlFlowToLLVMPass());

    // Finalization passes.
    passManager.addNestedPass<mlir::LLVM::LLVMFuncOp>(
        mlir::createReconcileUnrealizedCastsPass());

    passManager.addNestedPass<mlir::LLVM::LLVMFuncOp>(
        mlir::createCanonicalizerPass());

    passManager.addPass(mlir::LLVM::createLegalizeForExportPass());

    // If requested, print the statistics.
    if (ci.getFrontendOptions().printStatistics) {
      passManager.enableTiming();
      passManager.enableStatistics();
    }

    // Run the conversion.
    if (mlir::failed(passManager.run(ci.getMLIRModule()))) {
      ci.getDiagnostics().emitFatalError<GenericStringMessage>(
          "Modelica dialect conversion failure");
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

    // Convert to LLVM IR
    auto llvmModule = mlir::translateModuleToLLVMIR(ci.getMLIRModule(), ci.getLLVMContext());

    if (!llvmModule) {
      ci.getDiagnostics().emitFatalError<GenericStringMessage>("Failed to emit LLVM-IR");
      return false;
    }

    auto targetMachine = ci.getTargetMachine();

    if (!targetMachine) {
      return false;
    }

    llvmModule->setTargetTriple(ci.getCodegenOptions().target);
    llvmModule->setDataLayout(targetMachine->createDataLayout());

    // Optimize the IR
    auto optLevel = ci.getCodegenOptions().optLevel;
    auto optPipeline = mlir::makeOptimizingTransformer(optLevel.time, optLevel.size, nullptr);

    if (auto error = optPipeline(llvmModule.get())) {
      ci.getDiagnostics().emitError<GenericStringMessage>("Failed to optimize LLVM-IR");
      llvm::consumeError(std::move(error));
      return false;
    }

    instance().setLLVMModule(std::move(llvmModule));
    return true;
  }

  llvm::DataLayout FrontendAction::getDataLayout()
  {
    CompilerInstance& ci = instance();

    auto* targetMachine = ci.getTargetMachine();
    assert(targetMachine && "Can't instantiate the target machine");

    return targetMachine->createDataLayout();
  }

  std::string FrontendAction::getDataLayoutString()
  {
    std::string dataLayoutString = getDataLayout().getStringRepresentation();
    assert(dataLayoutString != "" && "Expecting a valid target data layout");

    return dataLayoutString;
  }

  std::unique_ptr<mlir::Pass>
  FrontendAction::createReadOnlyVariablesPropagationPass()
  {
    const CompilerInstance& ci = instance();

    mlir::modelica::ReadOnlyVariablesPropagationPassOptions options;
    options.modelName = ci.getSimulationOptions().modelName;

    return mlir::modelica::createReadOnlyVariablesPropagationPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createAutomaticDifferentiationPass()
  {
    return mlir::modelica::createAutomaticDifferentiationPass();
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createModelLegalizationPass()
  {
    const CompilerInstance& ci = instance();

    mlir::modelica::ModelLegalizationPassOptions options;
    options.modelName = ci.getSimulationOptions().modelName;

    return mlir::modelica::createModelLegalizationPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createMatchingPass()
  {
    const CompilerInstance& ci = instance();

    mlir::modelica::MatchingPassOptions options;
    options.modelName = ci.getSimulationOptions().modelName;

    return mlir::modelica::createMatchingPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createVariablesPromotionPass()
  {
    const CompilerInstance& ci = instance();

    mlir::modelica::VariablesPromotionPassOptions options;
    options.modelName = ci.getSimulationOptions().modelName;

    return mlir::modelica::createVariablesPromotionPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createCyclesSolvingPass()
  {
    const CompilerInstance& ci = instance();

    mlir::modelica::CyclesSolvingPassOptions options;
    options.modelName = ci.getSimulationOptions().modelName;
    options.allowUnsolvedCycles = ci.getSimulationOptions().solver == "ida";

    return mlir::modelica::createCyclesSolvingPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createSchedulingPass()
  {
    const CompilerInstance& ci = instance();

    mlir::modelica::SchedulingPassOptions options;
    options.modelName = ci.getSimulationOptions().modelName;

    return mlir::modelica::createSchedulingPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createEulerForwardPass()
  {
    const CompilerInstance& ci = instance();

    mlir::modelica::EulerForwardPassOptions options;
    options.bitWidth = ci.getCodegenOptions().bitWidth;
    options.dataLayout = getDataLayoutString();
    options.model = ci.getSimulationOptions().modelName;
    options.emitSimulationMainFunction = ci.getCodegenOptions().generateMain;
    options.variablesFilter = ci.getFrontendOptions().variablesFilter;

    return mlir::modelica::createEulerForwardPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createIDAPass()
  {
    const CompilerInstance& ci = instance();

    mlir::modelica::IDAPassOptions options;
    options.bitWidth = ci.getCodegenOptions().bitWidth;
    options.dataLayout = getDataLayoutString();
    options.model = ci.getSimulationOptions().modelName;
    options.emitSimulationMainFunction = ci.getCodegenOptions().generateMain;
    options.variablesFilter = ci.getFrontendOptions().variablesFilter;
    options.reducedSystem = ci.getSimulationOptions().IDAReducedSystem;
    options.reducedDerivatives = ci.getSimulationOptions().IDAReducedDerivatives;
    options.jacobianOneSweep = ci.getSimulationOptions().IDAJacobianOneSweep;

    return mlir::modelica::createIDAPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createFunctionScalarizationPass()
  {
    const CompilerInstance& ci = instance();

    mlir::modelica::FunctionScalarizationPassOptions options;
    options.assertions = ci.getCodegenOptions().assertions;
    return mlir::modelica::createFunctionScalarizationPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createExplicitCastInsertionPass()
  {
    return mlir::modelica::createExplicitCastInsertionPass();
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createModelicaToCFConversionPass()
  {
    const CompilerInstance& ci = instance();

    mlir::ModelicaToCFConversionPassOptions options;
    options.bitWidth = ci.getCodegenOptions().bitWidth;
    options.outputArraysPromotion = ci.getCodegenOptions().outputArraysPromotion;
    options.dataLayout = getDataLayoutString();

    return mlir::createModelicaToCFConversionPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createModelicaToVectorConversionPass()
  {
    const CompilerInstance& ci = instance();

    mlir::ModelicaToVectorConversionPassOptions options;
    options.bitWidth = ci.getCodegenOptions().bitWidth;

    return mlir::createModelicaToVectorConversionPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createModelicaToArithConversionPass()
  {
    const CompilerInstance& ci = instance();

    mlir::ModelicaToArithConversionPassOptions options;
    options.bitWidth = ci.getCodegenOptions().bitWidth;
    options.assertions = ci.getCodegenOptions().assertions;
    options.dataLayout = getDataLayoutString();

    return mlir::createModelicaToArithConversionPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createModelicaToFuncConversionPass()
  {
    const CompilerInstance& ci = instance();

    mlir::ModelicaToFuncConversionPassOptions options;
    options.bitWidth = ci.getCodegenOptions().bitWidth;
    options.dataLayout = getDataLayoutString();
    options.assertions = ci.getCodegenOptions().assertions;

    return mlir::createModelicaToFuncConversionPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createModelicaToMemRefConversionPass()
  {
    const CompilerInstance& ci = instance();

    mlir::ModelicaToMemRefConversionPassOptions options;
    options.bitWidth = ci.getCodegenOptions().bitWidth;
    options.assertions = ci.getCodegenOptions().assertions;
    options.dataLayout = getDataLayoutString();

    return mlir::createModelicaToMemRefConversionPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createModelicaToLLVMConversionPass()
  {
    const CompilerInstance& ci = instance();

    mlir::ModelicaToLLVMConversionPassOptions options;
    options.assertions = ci.getCodegenOptions().assertions;
    options.dataLayout = getDataLayoutString();

    return mlir::createModelicaToLLVMConversionPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createIDAToFuncConversionPass()
  {
    const CompilerInstance& ci = instance();

    mlir::IDAToFuncConversionPassOptions options;
    options.bitWidth = ci.getCodegenOptions().bitWidth;

    return mlir::createIDAToFuncConversionPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createIDAToLLVMConversionPass()
  {
    mlir::IDAToLLVMConversionPassOptions options;
    options.dataLayout = getDataLayoutString();

    return mlir::createIDAToLLVMConversionPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createKINSOLToLLVMConversionPass()
  {
    mlir::KINSOLToLLVMConversionPassOptions options;
    options.dataLayout = getDataLayoutString();

    return mlir::createKINSOLToLLVMConversionPass(options);
  }

  std::unique_ptr<mlir::Pass>
  FrontendAction::createSimulationToFuncConversionPass()
  {
    const CompilerInstance& ci = instance();

    mlir::SimulationToFuncConversionPassOptions options;
    options.bitWidth = ci.getCodegenOptions().bitWidth;

    return mlir::createSimulationToFuncConversionPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createFuncToLLVMConversionPass(
      bool useBarePtrCallConv)
  {
    CompilerInstance& ci = instance();

    mlir::LowerToLLVMOptions options(&ci.getMLIRContext());
    options.dataLayout = getDataLayout();
    options.useBarePtrCallConv = useBarePtrCallConv;

    return mlir::createConvertFuncToLLVMPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createArithToLLVMConversionPass()
  {
    mlir::ArithToLLVMConversionPassOptions options;
    return mlir::createArithToLLVMConversionPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createMemRefToLLVMConversionPass()
  {
    mlir::MemRefToLLVMConversionPassOptions options;
    options.useGenericFunctions = true;

    return mlir::createMemRefToLLVMConversionPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createVectorToLLVMConversionPass()
  {
    mlir::LowerVectorToLLVMOptions options;
    return mlir::createConvertVectorToLLVMPass(options);
  }

  std::unique_ptr<mlir::Pass> FrontendAction::createVectorToSCFConversionPass()
  {
    mlir::VectorTransferToSCFOptions options;
    return mlir::createConvertVectorToSCFPass(options);
  }
}
