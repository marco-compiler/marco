#include "marco/Frontend/FrontendActions.h"
#include "marco/Diagnostic/Printer.h"
#include "marco/Codegen/Bridge.h"
#include "marco/Codegen/Verifier.h"
#include "marco/Codegen/Conversion/Passes.h"
#include "marco/Codegen/Transforms/Passes.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Dialect/Modeling/ModelingDialect.h"
#include "marco/Dialect/Simulation/SimulationDialect.h"
#include "marco/Frontend/CompilerInstance.h"
#include "marco/IO/Command.h"
#include "marco/Parser/Parser.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/TargetParser/Host.h"

using namespace ::marco;
using namespace ::marco::diagnostic;
using namespace ::marco::frontend;
using namespace ::marco::io;

//===---------------------------------------------------------------------===//
// Messages
//===---------------------------------------------------------------------===//

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

  class InvalidTargetTripleMessage : public Message
  {
    public:
      InvalidTargetTripleMessage(llvm::StringRef targetTriple)
          : targetTriple(targetTriple.str())
      {
      }

      void print(PrinterInstance* printer) const override
      {
        auto& os = printer->getOutputStream();
        os << "Invalid target triple '" << targetTriple << "'\n";
      }

    private:
      std::string targetTriple;
  };
}

//===---------------------------------------------------------------------===//
// Utility functions
//===---------------------------------------------------------------------===//

static llvm::CodeGenOpt::Level mapOptimizationLevelToCodeGenLevel(
    llvm::OptimizationLevel level, bool debug)
{
  if (level == llvm::OptimizationLevel::O0) {
    return llvm::CodeGenOpt::Level::None;;
  }

  if (debug || level == llvm::OptimizationLevel::O1) {
    return llvm::CodeGenOpt::Level::Less;
  }

  if (level == llvm::OptimizationLevel::O2 ||
      level == llvm::OptimizationLevel::Os ||
      level == llvm::OptimizationLevel::Oz) {
    return llvm::CodeGenOpt::Level::Default;
  }

  assert(level == llvm::OptimizationLevel::O3);
  return llvm::CodeGenOpt::Level::Aggressive;
}

/// Generate target-specific machine-code or assembly file from the input LLVM
/// module.
///
/// @param diags        Diagnostics engine for reporting errors
/// @param tm           Target machine to aid the code-gen pipeline set-up
/// @param act          Backend act to run (assembly vs machine-code generation)
/// @param llvmModule   LLVM module to lower to assembly/machine-code
/// @param os           Output stream to emit the generated code to
static void generateMachineCodeOrAssemblyImpl(
    diagnostic::DiagnosticEngine& diags,
    llvm::TargetMachine& tm,
    llvm::CodeGenFileType fileType,
    llvm::Module& llvmModule,
    llvm::raw_pwrite_stream& os)
{
  // Set-up the pass manager, i.e create an LLVM code-gen pass pipeline.
  // Currently only the legacy pass manager is supported.
  // TODO: Switch to the new PM once it's available in the backend.
  llvm::legacy::PassManager codeGenPasses;

  codeGenPasses.add(
      llvm::createTargetTransformInfoWrapperPass(tm.getTargetIRAnalysis()));

  llvm::Triple triple(llvmModule.getTargetTriple());

  std::unique_ptr<llvm::TargetLibraryInfoImpl> tlii =
      std::make_unique<llvm::TargetLibraryInfoImpl>(triple);

  assert(tlii && "Failed to create TargetLibraryInfo");
  codeGenPasses.add(new llvm::TargetLibraryInfoWrapperPass(*tlii));

  if (tm.addPassesToEmitFile(codeGenPasses, os, nullptr, fileType)) {
    diags.emitFatalError<GenericStringMessage>(
        "TargetMachine can't emit a file of this type");

    return;
  }

  // Run the passes.
  codeGenPasses.run(llvmModule);

  os.flush();
}

namespace marco::frontend
{
  bool PreprocessingAction::beginSourceFilesAction()
  {
    CompilerInstance& ci = getInstance();

    if (ci.getFrontendOptions().omcBypass) {
      const auto& inputs = ci.getFrontendOptions().inputs;

      if (inputs.size() > 1) {
        ci.getDiagnostics().emitFatalError<GenericStringMessage>(
            "MARCO can receive only one input flattened file");

        return false;
      }

      auto errorOrBuffer = llvm::MemoryBuffer::getFileOrSTDIN(inputs[0].getFile());
      auto buffer = llvm::errorOrToExpected(std::move(errorOrBuffer));

      if (!buffer) {
        ci.getDiagnostics().emitFatalError<CantOpenInputFileMessage>(inputs[0].getFile());
        llvm::consumeError(buffer.takeError());
        return false;
      }

      flattened = (*buffer)->getBuffer().str();
      return true;
    }

    // Use OMC to generate the Base Modelica code.
    auto omcPath = llvm::sys::findProgramByName("omc");

    if (!omcPath) {
      ci.getDiagnostics().emitWarning<GenericStringMessage>(
          "Can't obtain path to omc executable");

      return false;
    }

    Command cmd(*omcPath);

    // Add the input files.
    for (const InputFile& inputFile : ci.getFrontendOptions().inputs) {
      if (inputFile.getKind().getLanguage() != Language::Modelica) {
        ci.getDiagnostics().emitError<GenericStringMessage>(
            "Invalid input type: flattening can be performed only on Modelica files");

        return false;
      }

      cmd.appendArg(inputFile.getFile().str());
    }

    // Set the model to be flattened.
    if (const auto& modelName = ci.getSimulationOptions().modelName;
        modelName.empty()) {
      ci.getDiagnostics().emitWarning<GenericStringMessage>(
          "Model name not specified");
    } else {
      cmd.appendArg("+i=" + ci.getSimulationOptions().modelName);
    }

    // Enable the output of base Modelica.
    cmd.appendArg("-f");

    // Add the extra arguments to OMC.
    for (const auto& arg : ci.getFrontendOptions().omcCustomArgs) {
      cmd.appendArg(arg);
    }

    // Create the file in which OMC will write.
    llvm::SmallVector<char, 128> tmpPath;
    llvm::sys::path::system_temp_directory(true, tmpPath);
    llvm::sys::path::append(tmpPath, "marco_omc_%%%%%%.bmo");

    llvm::Expected<llvm::sys::fs::TempFile> tempFile =
        llvm::sys::fs::TempFile::create(tmpPath);

    if (!tempFile) {
      ci.getDiagnostics().emitFatalError<GenericStringMessage>(
          "Can't create OMC output file");

      return false;
    }

    cmd.setStdoutRedirect(tempFile->TmpName);
    int resultCode = cmd.exec();

    if (resultCode == EXIT_SUCCESS) {
      auto errorOrBuffer = llvm::MemoryBuffer::getFileOrSTDIN(tempFile->TmpName);
      auto buffer = llvm::errorOrToExpected(std::move(errorOrBuffer));

      if (!buffer) {
        ci.getDiagnostics().emitFatalError<CantOpenInputFileMessage>(tempFile->TmpName);
        llvm::consumeError(buffer.takeError());
        return false;
      }

      flattened = (*buffer)->getBuffer().str();
    }

    if (auto ec = tempFile->discard()) {
      ci.getDiagnostics().emitFatalError<GenericStringMessage>(
          "Can't erase the temporary OMC output file");

      return false;
    }

    return resultCode == EXIT_SUCCESS;
  }

  void EmitBaseModelicaAction::executeAction()
  {
    CompilerInstance& ci = getInstance();
    std::unique_ptr<llvm::raw_pwrite_stream> os;

    // Get the default output stream, if none was specified.
    if (ci.isOutputStreamNull()) {
      if (!(os = ci.createDefaultOutputFile(
                false, ci.getSimulationOptions().modelName, "bmo"))) {
        return;
      }
    }

    *os << flattened;
  }

  ASTAction::ASTAction(ASTActionKind action)
      : action(action)
  {
  }

  bool ASTAction::beginSourceFilesAction()
  {
    if (!PreprocessingAction::beginSourceFilesAction()) {
      return false;
    }

    CompilerInstance& ci = getInstance();
    auto& diagnostics = ci.getDiagnostics();

    // Parse the source code.
    auto sourceFile = std::make_shared<marco::SourceFile>(
        "-", llvm::MemoryBuffer::getMemBuffer(flattened));

    parser::Parser parser(diagnostics, sourceFile);
    auto cls = parser.parseRoot();

    if (!cls.has_value()) {
      ci.getDiagnostics().emitFatalError<GenericStringMessage>(
          "AST generation failed");
      return false;
    }

    ast = std::move(*cls);
    return true;
  }

  EmitASTAction::EmitASTAction()
      : ASTAction(ASTActionKind::Parse)
  {
  }

  void EmitASTAction::executeAction()
  {
    CompilerInstance& ci = getInstance();
    std::unique_ptr<llvm::raw_pwrite_stream> os;

    // Get the default output stream, if none was specified.
    if (ci.isOutputStreamNull()) {
      if (!(os = ci.createDefaultOutputFile(
                false, ci.getSimulationOptions().modelName, "ast"))) {
        return;
      }
    }

    // Get the JSON object for the AST.
    llvm::json::Value json = ast->toJSON();

    // Print the JSON object to file.
    if (ci.isOutputStreamNull()) {
      *os << llvm::formatv("{0:2}", json);
    } else {
      ci.getOutputStream() << llvm::formatv("{0:2}", json);
    }
  }

  void InitOnlyAction::executeAction()
  {
    CompilerInstance& ci = getInstance();

    ci.getDiagnostics().emitWarning<GenericStringMessage>(
        "Use '--init-only' for testing purposes only");

    auto& os = llvm::outs();

    const CodegenOptions& codegenOptions = ci.getCodeGenOptions();
    printCategory(os, "Code generation");
    printOption(os, "Time optimization level", static_cast<long>(codegenOptions.optLevel.getSpeedupLevel()));
    printOption(os, "Size optimization level", static_cast<long>(codegenOptions.optLevel.getSizeLevel()));
    printOption(os, "Debug information", codegenOptions.debug);
    printOption(os, "Assertions", codegenOptions.assertions);
    printOption(os, "Inlining", codegenOptions.inlining);
    printOption(os, "Output arrays promotion", codegenOptions.outputArraysPromotion);
    printOption(os, "Read-only variables propagation", codegenOptions.readOnlyVariablesPropagation);
    printOption(os, "CSE", codegenOptions.cse);
    printOption(os, "OpenMP", codegenOptions.omp);
    printOption(os, "Target triple", codegenOptions.target);
    printOption(os, "Target cpu", codegenOptions.cpu);
    os << "\n";

    const SimulationOptions& simulationOptions = ci.getSimulationOptions();
    printCategory(os, "Simulation");
    printOption(os, "Model", simulationOptions.modelName);
    printOption(os, "Solver", simulationOptions.solver);
    os << "\n";
  }

  void InitOnlyAction::printCategory(llvm::raw_ostream& os, llvm::StringRef category) const
  {
    os << "[" << category << "]\n";
  }

  void InitOnlyAction::printOption(llvm::raw_ostream& os, llvm::StringRef name, llvm::StringRef value)
  {
    os << " - " << name << ": " << value << "\n";
  }

  void InitOnlyAction::printOption(llvm::raw_ostream& os, llvm::StringRef name, bool value)
  {
    os << " - " << name << ": " << (value ? "true" : "false") << "\n";
  }

  void InitOnlyAction::printOption(llvm::raw_ostream& os, llvm::StringRef name, long value)
  {
    os << " - " << name << ": " << value << "\n";
  }

  void InitOnlyAction::printOption(llvm::raw_ostream& os, llvm::StringRef name, double value)
  {
    os << " - " << name << ": " << value << "\n";
  }

  CodeGenAction::CodeGenAction(CodeGenActionKind action)
      : ASTAction(ASTActionKind::Parse),
        action(action)
  {
  }

  CodeGenAction::~CodeGenAction() = default;

  bool CodeGenAction::beginSourceFilesAction()
  {
    if (action == CodeGenActionKind::GenerateLLVMIR) {
      return generateLLVMIR();
    }

    assert(action == CodeGenActionKind::GenerateMLIR);
    return generateMLIR();
  }

  bool CodeGenAction::setUpTargetMachine()
  {
    CompilerInstance& ci = getInstance();
    const std::string& triple = ci.getCodeGenOptions().target;

    // Get the LLVM target.
    std::string targetError;

    const llvm::Target* target =
        llvm::TargetRegistry::lookupTarget(triple, targetError);

    if (!target) {
      // Print an error and exit if we couldn't find the requested target.
      // This generally occurs if we've forgotten to initialize the
      // TargetRegistry or if we have a bogus target triple.

      ci.getDiagnostics().emitFatalError<InvalidTargetTripleMessage>(triple);
      return false;
    }

    // Create the LLVM TargetMachine.
    const CodegenOptions& codegenOptions = ci.getCodeGenOptions();

    llvm::CodeGenOpt::Level optLevel = mapOptimizationLevelToCodeGenLevel(
        codegenOptions.optLevel, codegenOptions.debug);

    std::string features = "";

    auto relocationModel =
        std::optional<llvm::Reloc::Model>(llvm::Reloc::PIC_);

    targetMachine.reset(target->createTargetMachine(
        triple, ci.getCodeGenOptions().cpu,
        features, llvm::TargetOptions(),
        relocationModel,
        std::nullopt, optLevel));

    if (!targetMachine) {
      ci.getDiagnostics().emitFatalError<GenericStringMessage>(
          "Can't create TargetMachine");

      return false;
    }

    return true;
  }

  llvm::DataLayout CodeGenAction::getDataLayout() const
  {
    assert(targetMachine && "TargetMachine has not been initialized yet");
    return targetMachine->createDataLayout();
  }

  void CodeGenAction::registerMLIRDialects()
  {
    // Register all the MLIR native dialects. This does not impact performance,
    // because of lazy loading.
    mlir::registerAllDialects(mlirDialectRegistry);

    // Register the custom dialects.
    mlirDialectRegistry.insert<mlir::modeling::ModelingDialect>();
    mlirDialectRegistry.insert<mlir::modelica::ModelicaDialect>();
    mlirDialectRegistry.insert<mlir::ida::IDADialect>();
    mlirDialectRegistry.insert<mlir::simulation::SimulationDialect>();
  }

  bool CodeGenAction::generateMLIR()
  {
    CompilerInstance& ci = getInstance();

    // Create the MLIR context.
    mlir::func::registerInlinerExtension(mlirDialectRegistry);

    mlirContext = std::make_unique<mlir::MLIRContext>(mlirDialectRegistry);
    mlirContext->enableMultithreading(ci.getFrontendOptions().multithreading);

    mlirContext->loadDialect<mlir::modelica::ModelicaDialect>();
    mlirContext->loadDialect<mlir::DLTIDialect>();
    mlirContext->loadDialect<mlir::LLVM::LLVMDialect>();

    if (getCurrentInputs()[0].getKind().getLanguage() == Language::MLIR) {
      // If the input is an MLIR file, parse it.
      llvm::SourceMgr sourceMgr;

      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
          llvm::MemoryBuffer::getFileOrSTDIN(getCurrentInputs()[0].getFile());

      sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

      // Register the dialects that can be parsed.
      registerMLIRDialects();

      // Parse the module.
      mlir::OwningOpRef<mlir::ModuleOp> module =
          mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, mlirContext.get());

      if (!module || mlir::failed(module->verifyInvariants())) {
        ci.getDiagnostics().emitError<GenericStringMessage>(
            "Could not parse MLIR");

        return false;
      }

      mlirModule = std::make_unique<mlir::ModuleOp>(module.release());
      return true;
    } else {
      // If the input is not an MLIR file, then the MLIR module will be
      // obtained starting from the AST.

      if (!ASTAction::beginSourceFilesAction()) {
        return false;
      }

      // Convert the AST to MLIR.
      codegen::CodegenOptions options;
      marco::codegen::lowering::Bridge bridge(*mlirContext, options);
      bridge.lower(*ast->cast<ast::Root>());

      mlirModule = std::move(bridge.getMLIRModule());
    }

    // Set target triple and data layout inside the MLIR module.
    setMLIRModuleTargetTriple();
    setMLIRModuleDataLayout();

    // Verify the IR.
    mlir::PassManager pm(mlirContext.get());
    pm.addPass(std::make_unique<codegen::lowering::VerifierPass>());

    if (!mlir::succeeded(pm.run(*mlirModule))) {
      ci.getDiagnostics().emitError<GenericStringMessage>(
          "Verification of MLIR code failed");

      return false;
    }

    return true;
  }

  void CodeGenAction::setMLIRModuleTargetTriple()
  {
    assert(mlirModule && "MLIR module has not been created yet");

    if (!targetMachine) {
      setUpTargetMachine();
    }

    const std::string& triple = targetMachine->getTargetTriple().str();

    llvm::StringRef tripleAttrName =
        mlir::LLVM::LLVMDialect::getTargetTripleAttrName();

    auto tripleAttr = mlir::StringAttr::get(mlirModule->getContext(), triple);

    auto existingTripleAttr =
        (*mlirModule)->getAttrOfType<mlir::StringAttr>(tripleAttrName);

    if (existingTripleAttr && existingTripleAttr != tripleAttr) {
      // The LLVM module already has a target triple which is different from
      // the one specified through the command-line.
      getInstance().getDiagnostics().emitWarning<GenericStringMessage>(
          "The target triple is being overridden");
    }

    (*mlirModule)->setAttr(tripleAttrName, tripleAttr);
  }

  void CodeGenAction::setMLIRModuleDataLayout()
  {
    assert(mlirModule && "MLIR module has not been created yet");

    if (!targetMachine) {
      setUpTargetMachine();
    }

    const llvm::DataLayout& dl = targetMachine->createDataLayout();

    llvm::StringRef dlAttrName =
        mlir::LLVM::LLVMDialect::getDataLayoutAttrName();

    auto dlAttr = mlir::StringAttr::get(
        mlirContext.get(), dl.getStringRepresentation());

    auto existingDlAttr =
        (*mlirModule)->getAttrOfType<mlir::StringAttr>(dlAttrName);

    llvm::StringRef dlSpecAttrName =
        mlir::DLTIDialect::kDataLayoutAttrName;

    mlir::DataLayoutSpecInterface dlSpecAttr =
        mlir::translateDataLayout(dl, mlirContext.get());

    auto existingDlSpecAttr =
        (*mlirModule)->getAttrOfType<mlir::DataLayoutSpecInterface>(
            dlSpecAttrName);

    if ((existingDlAttr && existingDlAttr != dlAttr) ||
        (existingDlSpecAttr && existingDlSpecAttr != dlSpecAttr)) {
      // The MLIR module already has a data layout which is different from the
      // one given by the target machine.
      getInstance().getDiagnostics().emitWarning<GenericStringMessage>(
          "The data layout is being overridden");
    }

    (*mlirModule)->setAttr(dlAttrName, dlAttr);
    (*mlirModule)->setAttr(dlSpecAttrName, dlSpecAttr);
  }

  void CodeGenAction::createModelicaToLLVMPassPipeline(mlir::PassManager& pm)
  {
    CompilerInstance& ci = getInstance();

    if (!ci.getCodeGenOptions().debug) {
      // Remove the debug information if a non-debuggable executable has been
      // requested. By doing this at the beginning of the compilation pipeline
      // we reduce the time needed for the pass itself, as the code inevitably
      // grows while traversing the pipeline.
      pm.addPass(mlir::createStripDebugInfoPass());
    }

    // Try to simplify the starting IR.
    pm.addPass(mlir::createCanonicalizerPass());

    if (ci.getCodeGenOptions().readOnlyVariablesPropagation) {
      // Propagate the read-only variables and try to fold the constants right
      // after.
      pm.addPass(createMLIRReadOnlyVariablesPropagationPass());
      pm.addPass(mlir::createCanonicalizerPass());
    }

    pm.addPass(mlir::modelica::createAutomaticDifferentiationPass());

    // Inline the functions marked as "inlinable", in order to enable
    // simplifications for the model solving process.
    pm.addPass(mlir::modelica::createFunctionInliningPass());

    // Unpack the records.
    pm.addPass(mlir::modelica::createRecordInliningPass());
    pm.addPass(mlir::createCanonicalizerPass());

    // Infer the range boundaries for subscriptions.
    pm.addPass(mlir::modelica::createRangeBoundariesInferencePass());
    pm.addPass(mlir::createCanonicalizerPass());

    // Lift the equations.
    pm.addNestedPass<mlir::modelica::ModelOp>(
        mlir::modelica::createEquationTemplatesCreationPass());

    pm.addNestedPass<mlir::modelica::ModelOp>(
        mlir::modelica::createEquationViewsComputationPass());

    // Handle the derivatives.
    pm.addNestedPass<mlir::modelica::ModelOp>(
        mlir::modelica::createDerivativesAllocationPass());

    // Legalize the model.
    pm.addNestedPass<mlir::modelica::ModelOp>(
        mlir::modelica::createBindingEquationConversionPass());

    pm.addNestedPass<mlir::modelica::ModelOp>(
        mlir::modelica::createExplicitStartValueInsertionPass());

    pm.addPass(mlir::modelica::createModelAlgorithmConversionPass());

    pm.addNestedPass<mlir::modelica::ModelOp>(
        mlir::modelica::createExplicitInitialEquationsInsertionPass());

    // Solve the model.
    pm.addNestedPass<mlir::modelica::ModelOp>(
        mlir::modelica::createMatchingPass());

    pm.addNestedPass<mlir::modelica::ModelOp>(
        mlir::modelica::createEquationAccessSplitPass());

    pm.addNestedPass<mlir::modelica::ModelOp>(
        mlir::modelica::createVariablesPromotionPass());

    pm.addNestedPass<mlir::modelica::ModelOp>(createMLIRCyclesSolvingPass());

    pm.addNestedPass<mlir::modelica::ModelOp>(
        mlir::modelica::createSchedulingPass());

    // Apply the selected solver.
    pm.addPass(
        llvm::StringSwitch<std::unique_ptr<mlir::Pass>>(
            ci.getSimulationOptions().solver)
            .Case("euler-forward", createMLIREulerForwardPass())
            .Case("ida", createMLIRIDAPass())
            .Default(createMLIREulerForwardPass()));

    pm.addPass(createMLIRFunctionScalarizationPass());
    pm.addPass(mlir::modelica::createExplicitCastInsertionPass());
    pm.addPass(mlir::createCanonicalizerPass());

    if (ci.getCodeGenOptions().cse) {
      pm.addNestedPass<mlir::modelica::FunctionOp>(
          mlir::createCSEPass());

      pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
    }

    pm.addPass(mlir::modelica::createArrayDeallocationPass());
    pm.addPass(createMLIRModelicaToCFConversionPass());

    if (ci.getCodeGenOptions().inlining) {
      // Inline the functions with the 'inline' annotation.
      pm.addPass(mlir::createInlinerPass());
    }

    pm.addPass(createMLIRModelicaToVectorConversionPass());
    pm.addPass(createMLIRModelicaToArithConversionPass());
    pm.addPass(createMLIRModelicaToFuncConversionPass());
    pm.addPass(createMLIRModelicaToMemRefConversionPass());

    pm.addPass(mlir::createSUNDIALSToFuncConversionPass());
    pm.addPass(createMLIRIDAToFuncConversionPass());

    if (ci.getCodeGenOptions().omp) {
      // Use OpenMP for parallel loops.
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::createConvertSCFToOpenMPPass());
    }

    // Try to fold constants and run again the Modelica -> Arith conversion
    // pass to convert the new constants.
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(createMLIRModelicaToArithConversionPass());

    pm.addNestedPass<mlir::func::FuncOp>(
        createMLIRVectorToSCFConversionPass());

    pm.addPass(createMLIRVectorToLLVMConversionPass());
    pm.addPass(createMLIRArithToLLVMConversionPass());
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(createMLIRArithToLLVMConversionPass());
    pm.addPass(createMLIRMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertSCFToCFPass());

    pm.addPass(createMLIRFuncToLLVMConversionPass(true));
    pm.addPass(createMLIRFuncToLLVMConversionPass(false));

    pm.addPass(mlir::createConvertControlFlowToLLVMPass());

    // Convert the MARCO dialects to LLVM dialect.
    pm.addPass(createMLIRModelicaToLLVMConversionPass());
    pm.addPass(createMLIRIDAToLLVMConversionPass());

    // Now that the Simulation dialect doesn't have dependencies from Modelica
    // or the solvers, we can proceed converting it.
    pm.addPass(createMLIRSimulationToFuncConversionPass());

    // Convert the non-LLVM operations that may have been introduced by the
    // last conversions.
    pm.addNestedPass<mlir::func::FuncOp>(
        createMLIRArithToLLVMConversionPass());

    pm.addPass(createMLIRFuncToLLVMConversionPass(false));
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());

    // Finalization passes.
    pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(
        mlir::createReconcileUnrealizedCastsPass());

    pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(
        mlir::createCanonicalizerPass());

    pm.addPass(mlir::LLVM::createLegalizeForExportPass());

    // If requested, print the statistics.
    if (ci.getFrontendOptions().printStatistics) {
      pm.enableTiming();
      pm.enableStatistics();
    }
  }

  std::unique_ptr<mlir::Pass>
  CodeGenAction::createMLIRFunctionScalarizationPass()
  {
    CompilerInstance& ci = getInstance();

    mlir::modelica::FunctionScalarizationPassOptions options;
    options.assertions = ci.getCodeGenOptions().assertions;

    return mlir::modelica::createFunctionScalarizationPass(options);
  }

  std::unique_ptr<mlir::Pass>
  CodeGenAction::createMLIRReadOnlyVariablesPropagationPass()
  {
    CompilerInstance& ci = getInstance();

    mlir::modelica::ReadOnlyVariablesPropagationPassOptions options;
    options.modelName = ci.getSimulationOptions().modelName;

    return mlir::modelica::createReadOnlyVariablesPropagationPass(options);
  }

  std::unique_ptr<mlir::Pass> CodeGenAction::createMLIRCyclesSolvingPass()
  {
    CompilerInstance& ci = getInstance();

    mlir::modelica::CyclesSolvingPassOptions options;
    options.allowUnsolvedCycles = ci.getSimulationOptions().solver == "ida";

    return mlir::modelica::createCyclesSolvingPass(options);
  }

  std::unique_ptr<mlir::Pass> CodeGenAction::createMLIREulerForwardPass()
  {
    CompilerInstance& ci = getInstance();

    mlir::modelica::EulerForwardPassOptions options;
    options.variablesFilter = ci.getFrontendOptions().variablesFilter;

    return mlir::modelica::createEulerForwardPass(options);
  }

  std::unique_ptr<mlir::Pass> CodeGenAction::createMLIRIDAPass()
  {
    CompilerInstance& ci = getInstance();

    mlir::modelica::IDAPassOptions options;
    options.variablesFilter = ci.getFrontendOptions().variablesFilter;
    options.reducedSystem = ci.getSimulationOptions().IDAReducedSystem;
    options.reducedDerivatives = ci.getSimulationOptions().IDAReducedDerivatives;
    options.jacobianOneSweep = ci.getSimulationOptions().IDAJacobianOneSweep;
    options.debugInformation = ci.getCodeGenOptions().debug;

    return mlir::modelica::createIDAPass(options);
  }

  std::unique_ptr<mlir::Pass>
  CodeGenAction::createMLIRModelicaToArithConversionPass()
  {
    CompilerInstance& ci = getInstance();

    mlir::ModelicaToArithConversionPassOptions options;
    options.bitWidth = ci.getCodeGenOptions().bitWidth;
    options.assertions = ci.getCodeGenOptions().assertions;
    options.dataLayout = getDataLayout().getStringRepresentation();

    return mlir::createModelicaToArithConversionPass(options);
  }

  std::unique_ptr<mlir::Pass>
  CodeGenAction::createMLIRModelicaToCFConversionPass()
  {
    CompilerInstance& ci = getInstance();

    mlir::ModelicaToCFConversionPassOptions options;
    options.bitWidth = ci.getCodeGenOptions().bitWidth;

    options.outputArraysPromotion =
        ci.getCodeGenOptions().outputArraysPromotion;

    options.dataLayout = getDataLayout().getStringRepresentation();

    return mlir::createModelicaToCFConversionPass(options);
  }

  std::unique_ptr<mlir::Pass>
  CodeGenAction::createMLIRModelicaToFuncConversionPass()
  {
    CompilerInstance& ci = getInstance();

    mlir::ModelicaToFuncConversionPassOptions options;
    options.bitWidth = ci.getCodeGenOptions().bitWidth;
    options.dataLayout = getDataLayout().getStringRepresentation();
    options.assertions = ci.getCodeGenOptions().assertions;

    return mlir::createModelicaToFuncConversionPass(options);
  }

  std::unique_ptr<mlir::Pass>
  CodeGenAction::createMLIRModelicaToLLVMConversionPass()
  {
    CompilerInstance& ci = getInstance();

    mlir::ModelicaToLLVMConversionPassOptions options;
    options.assertions = ci.getCodeGenOptions().assertions;
    options.dataLayout = getDataLayout().getStringRepresentation();

    return mlir::createModelicaToLLVMConversionPass(options);
  }

  std::unique_ptr<mlir::Pass>
  CodeGenAction::createMLIRModelicaToMemRefConversionPass()
  {
    CompilerInstance& ci = getInstance();

    mlir::ModelicaToMemRefConversionPassOptions options;
    options.bitWidth = ci.getCodeGenOptions().bitWidth;
    options.assertions = ci.getCodeGenOptions().assertions;
    options.dataLayout = getDataLayout().getStringRepresentation();

    return mlir::createModelicaToMemRefConversionPass(options);
  }

  std::unique_ptr<mlir::Pass>
  CodeGenAction::createMLIRModelicaToVectorConversionPass()
  {
    CompilerInstance& ci = getInstance();

    mlir::ModelicaToVectorConversionPassOptions options;
    options.bitWidth = ci.getCodeGenOptions().bitWidth;

    return mlir::createModelicaToVectorConversionPass(options);
  }

  std::unique_ptr<mlir::Pass>
  CodeGenAction::createMLIRIDAToFuncConversionPass()
  {
    CompilerInstance& ci = getInstance();

    mlir::IDAToFuncConversionPassOptions options;
    options.bitWidth = ci.getCodeGenOptions().bitWidth;

    return mlir::createIDAToFuncConversionPass(options);
  }

  std::unique_ptr<mlir::Pass>
  CodeGenAction::createMLIRIDAToLLVMConversionPass()
  {
    mlir::IDAToLLVMConversionPassOptions options;
    options.dataLayout = getDataLayout().getStringRepresentation();

    return mlir::createIDAToLLVMConversionPass(options);
  }

  std::unique_ptr<mlir::Pass>
  CodeGenAction::createMLIRSimulationToFuncConversionPass()
  {
    return mlir::createSimulationToFuncConversionPass();
  }

  std::unique_ptr<mlir::Pass>
  CodeGenAction::createMLIRArithToLLVMConversionPass()
  {
    mlir::ArithToLLVMConversionPassOptions options;
    return mlir::createArithToLLVMConversionPass(options);
  }

  std::unique_ptr<mlir::Pass>
  CodeGenAction::createMLIRFuncToLLVMConversionPass(bool useBarePtrCallConv)
  {
    mlir::ConvertFuncToLLVMPassOptions options;
    options.useBarePtrCallConv = useBarePtrCallConv;

    return mlir::createConvertFuncToLLVMPass(options);
  }

  std::unique_ptr<mlir::Pass>
  CodeGenAction::createMLIRMemRefToLLVMConversionPass()
  {
    mlir::FinalizeMemRefToLLVMConversionPassOptions options;
    options.useGenericFunctions = true;

    return mlir::createFinalizeMemRefToLLVMConversionPass(options);
  }

  std::unique_ptr<mlir::Pass>
  CodeGenAction::createMLIRVectorToLLVMConversionPass()
  {
    mlir::ConvertVectorToLLVMPassOptions options;
    return mlir::createConvertVectorToLLVMPass(options);
  }

  std::unique_ptr<mlir::Pass>
  CodeGenAction::createMLIRVectorToSCFConversionPass()
  {
    mlir::VectorTransferToSCFOptions options;
    return mlir::createConvertVectorToSCFPass(options);
  }

  void CodeGenAction::registerMLIRToLLVMIRTranslations()
  {
    assert(mlirContext && "MLIR context has not been created yet");
    mlir::registerBuiltinDialectTranslation(*mlirContext);
    mlir::registerLLVMDialectTranslation(*mlirContext);
    mlir::registerOpenMPDialectTranslation(*mlirContext);
  }

  bool CodeGenAction::generateLLVMIR()
  {
    CompilerInstance& ci = getInstance();

    // Create the LLVM context.
    llvmContext = std::make_unique<llvm::LLVMContext>();

    if (getCurrentInputs()[0].getKind().getLanguage() == Language::LLVM_IR) {
      // If the input is an LLVM file, parse it return.
      llvm::SMDiagnostic err;

      llvmModule = llvm::parseIRFile(
          getCurrentInputs()[0].getFile(), err, *llvmContext);

      if (!llvmModule || llvm::verifyModule(*llvmModule, &llvm::errs())) {
        err.print("marco", llvm::errs());

        ci.getDiagnostics().emitError<GenericStringMessage>(
            "Could not parse LLVM-IR");

        return false;
      }
    } else {
      // If the input is not an LLVM file, then the LLVM module will be
      // obtained starting from the MLIR module.

      if (!generateMLIR()) {
        return false;
      }

      // Register the MLIR translations to obtain LLVM-IR.
      registerMLIRToLLVMIRTranslations();

      // Set up the MLIR pass manager.
      mlir::PassManager pm(mlirContext.get());
      pm.enableVerifier(true);

      // Create the pass pipeline.
      createModelicaToLLVMPassPipeline(pm);

      // Run the pass manager.
      if (!mlir::succeeded(pm.run(*mlirModule))) {
        ci.getDiagnostics().emitError<GenericStringMessage>(
            "Lower to LLVM dialect failed");

        return false;
      }

      // Translate to LLVM-IR.
      std::optional<llvm::StringRef> moduleName = mlirModule->getName();

      if (!moduleName) {
        // Fallback to the name of the compiled model.
        if (llvm::StringRef modelName = ci.getSimulationOptions().modelName;
            !modelName.empty()) {
          moduleName = modelName;
        }
      }

      llvmModule = mlir::translateModuleToLLVMIR(
          *mlirModule, *llvmContext,
          moduleName ? *moduleName : "ModelicaModule");

      if (!llvmModule) {
        ci.getDiagnostics().emitError<GenericStringMessage>(
            "Failed to create the LLVM module");

        return false;
      }
    }

    // Set target triple and data layout inside the LLVM module.
    setLLVMModuleTargetTriple();
    setLLVMModuleDataLayout();

    // Apply the optimizations to the LLVM module.
    runOptimizationPipeline();

    return true;
  }

  void CodeGenAction::setLLVMModuleTargetTriple()
  {
    assert(llvmModule && "LLVM module has not been created yet");

    if (!targetMachine) {
      setUpTargetMachine();
    }

    const std::string& triple = targetMachine->getTargetTriple().str();

    if (llvmModule->getTargetTriple() != triple) {
      // The LLVM module already has a target triple which is different from
      // the one specified through the command-line.
      getInstance().getDiagnostics().emitWarning<GenericStringMessage>(
          "The target triple is being overridden");
    }

    llvmModule->setTargetTriple(triple);
  }

  void CodeGenAction::setLLVMModuleDataLayout()
  {
    assert(llvmModule && "LLVM module has not been created yet");

    if (!targetMachine) {
      setUpTargetMachine();
    }

    const llvm::DataLayout& dataLayout = targetMachine->createDataLayout();

    if (llvmModule->getDataLayout() != dataLayout) {
      // The LLVM module already has a data layout which is different from the
      // one given by the target machine.
      getInstance().getDiagnostics().emitWarning<GenericStringMessage>(
          "The data layout is being overridden");
    }

    llvmModule->setDataLayout(dataLayout);
  }

  void CodeGenAction::runOptimizationPipeline()
  {
    CompilerInstance& ci = getInstance();
    const CodegenOptions& codegenOptions = ci.getCodeGenOptions();

    // Create the analysis managers.
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    // Create the pass manager builder.
    llvm::PassInstrumentationCallbacks pic;
    llvm::PipelineTuningOptions pto;
    std::optional<llvm::PGOOptions> pgoOpt;

    llvm::StandardInstrumentations si(*llvmContext, false);
    si.registerCallbacks(pic, &mam);

    llvm::PassBuilder pb(targetMachine.get(), pto, pgoOpt, &pic);

    // Register all the basic analyses with the managers.
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    // Create the pass manager.
    llvm::ModulePassManager mpm;

    if (codegenOptions.optLevel == llvm::OptimizationLevel::O0) {
      mpm = pb.buildO0DefaultPipeline(codegenOptions.optLevel, false);
    } else {
      mpm = pb.buildPerModuleDefaultPipeline(codegenOptions.optLevel);
    }

    // Run the passes.
    mpm.run(*llvmModule, mam);
  }

  EmitMLIRAction::EmitMLIRAction()
      : CodeGenAction(CodeGenActionKind::GenerateMLIR)
  {
  }

  void EmitMLIRAction::executeAction()
  {
    CompilerInstance& ci = getInstance();
    std::unique_ptr<llvm::raw_pwrite_stream> os;

    // Get the default output stream, if none was specified.
    if (ci.isOutputStreamNull()) {
      auto fileOrBufferNames = getCurrentFilesOrBufferNames();

      if (!(os = ci.createDefaultOutputFile(
                false, ci.getSimulationOptions().modelName, "mlir"))) {
        return;
      }
    }

    // Emit MLIR.
    mlirModule->print(ci.isOutputStreamNull() ? *os : ci.getOutputStream());
  }

  EmitLLVMIRAction::EmitLLVMIRAction()
      : CodeGenAction(CodeGenActionKind::GenerateLLVMIR)
  {
  }

  void EmitLLVMIRAction::executeAction()
  {
    CompilerInstance& ci = getInstance();

    // Get the default output stream, if none was specified.
    std::unique_ptr<llvm::raw_pwrite_stream> os;

    if (ci.isOutputStreamNull()) {
      if (!(os = ci.createDefaultOutputFile(
                false, ci.getSimulationOptions().modelName, "ll"))) {
        return;
      }
    }

    // Emit LLVM-IR.
    llvmModule->print(
        ci.isOutputStreamNull() ? *os : ci.getOutputStream(), nullptr);
  }

  EmitBitcodeAction::EmitBitcodeAction()
      : CodeGenAction(CodeGenActionKind::GenerateLLVMIR)
  {
  }

  void EmitBitcodeAction::executeAction()
  {
    CompilerInstance& ci = getInstance();

    // Get the default output stream, if none was specified.
    std::unique_ptr<llvm::raw_pwrite_stream> os;

    if (ci.isOutputStreamNull()) {
      if (!(os = ci.createDefaultOutputFile(
                true, ci.getSimulationOptions().modelName, "bc"))) {
        return;
      }
    }

    // Emit the bitcode.
    llvm::PassBuilder pb(targetMachine.get());

    llvm::ModuleAnalysisManager mam;
    pb.registerModuleAnalyses(mam);

    llvm::ModulePassManager mpm;

    mpm.addPass(llvm::BitcodeWriterPass(
        ci.isOutputStreamNull() ? *os : ci.getOutputStream()));

    mpm.run(*llvmModule, mam);
  }

  EmitAssemblyAction::EmitAssemblyAction()
      : CodeGenAction(CodeGenActionKind::GenerateLLVMIR)
  {
  }

  void EmitAssemblyAction::executeAction()
  {
    CompilerInstance& ci = getInstance();

    // Get the default output stream, if none was specified.
    std::unique_ptr<llvm::raw_pwrite_stream> os;

    if (ci.isOutputStreamNull()) {
      if (!(os = ci.createDefaultOutputFile(
                false, ci.getSimulationOptions().modelName, "s"))) {
        return;
      }
    }

    // Emit the assembly code.
    generateMachineCodeOrAssemblyImpl(
        ci.getDiagnostics(), *targetMachine,
        llvm::CodeGenFileType::CGFT_AssemblyFile, *llvmModule,
        ci.isOutputStreamNull() ? *os : ci.getOutputStream());
  }

  EmitObjAction::EmitObjAction()
      : CodeGenAction(CodeGenActionKind::GenerateLLVMIR)
  {
  }

  void EmitObjAction::executeAction()
  {
    CompilerInstance& ci = getInstance();

    // Get the default output stream, if none was specified.
    std::unique_ptr<llvm::raw_pwrite_stream> os;

    if (ci.isOutputStreamNull()) {
      if (!(os = ci.createDefaultOutputFile(
                true, ci.getSimulationOptions().modelName, "o"))) {
        return;
      }
    }

    // Emit the object file.
    generateMachineCodeOrAssemblyImpl(
        ci.getDiagnostics(), *targetMachine,
        llvm::CodeGenFileType::CGFT_ObjectFile, *llvmModule,
        ci.isOutputStreamNull() ? *os : ci.getOutputStream());
  }
}
