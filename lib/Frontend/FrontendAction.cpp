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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

    if (!cls.hasValue()) {
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
    auto& codegenOptions = ci.getCodegenOptions();
    auto& simulationOptions = ci.getSimulationOptions();

    // Set the target triple inside the MLIR module
    instance().getMLIRModule()->setAttr(
        mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
        mlir::StringAttr::get(&instance().getMLIRContext(), codegenOptions.target));

    // Set the data layout inside the MLIR module
    auto* targetMachine = ci.getTargetMachine();

    if (!targetMachine) {
      return false;
    }

    const llvm::DataLayout& dataLayout = targetMachine->createDataLayout();
    std::string dataLayoutString = dataLayout.getStringRepresentation();
    assert(dataLayoutString != "" && "Expecting a valid target data layout");

    instance().getMLIRModule()->setAttr(
        mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
        mlir::StringAttr::get(&instance().getMLIRContext(), dataLayoutString));

    // Create the pass manager and populate it with the appropriate transformations
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
    modelicaToCFOptions.dataLayout = dataLayout;

    passManager.addPass(codegen::createModelicaToCFPass(modelicaToCFOptions));

    // Modelica to LLVM conversion
    codegen::ModelicaToLLVMOptions modelicaToLLVMOptions;
    modelicaToLLVMOptions.assertions = codegenOptions.assertions;
    modelicaToLLVMOptions.dataLayout = dataLayout;

    passManager.addPass(codegen::createModelicaToLLVMPass(modelicaToLLVMOptions));

    // IDA to LLVM conversion
    codegen::IDAToLLVMOptions idaToLLVMOptions;
    idaToLLVMOptions.dataLayout = dataLayout;
    passManager.addPass(codegen::createIDAToLLVMPass());

    if (codegenOptions.omp) {
      // Use OpenMP for parallel loops
      passManager.addNestedPass<mlir::func::FuncOp>(mlir::createConvertSCFToOpenMPPass());
    }

    passManager.addPass(mlir::arith::createConvertArithmeticToLLVMPass());
    passManager.addPass(mlir::createConvertSCFToCFPass());

    mlir::LowerToLLVMOptions funcToLLVMOptions(&ci.getMLIRContext());
    funcToLLVMOptions.dataLayout = dataLayout;
    passManager.addPass(mlir::createConvertFuncToLLVMPass(funcToLLVMOptions));

    passManager.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (!codegenOptions.debug) {
      // Remove the debug information if a non-debuggable executable has been requested
      passManager.addPass(mlir::createStripDebugInfoPass());
    }

    // Run the conversion
    if (auto status = passManager.run(ci.getMLIRModule()); mlir::failed(status)) {
      ci.getDiagnostics().emitFatalError<GenericStringMessage>("Modelica dialect conversion failure");
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
}
