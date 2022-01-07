#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/Utils.h>
#include <marco/frontend/CompilerInstance.h>
#include <marco/frontend/FrontendAction.h>
#include <marco/frontend/FrontendActions.h>
#include <marco/frontend/FrontendOptions.h>
#include <marco/codegen/CodeGen.h>
#include <clang/Basic/DiagnosticFrontend.h>
#include <llvm/Support/Errc.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <marco/ast/Parser.h>
#include <marco/ast/Passes.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

namespace marco::frontend
{
  bool FrontendAction::runParse()
  {
    instance().classes().clear();

    for (const auto& input : instance().frontendOpts().inputs) {
      auto errorOrBuffer = llvm::MemoryBuffer::getFileOrSTDIN(input.file());
      auto buffer = llvm::errorOrToExpected(std::move(errorOrBuffer));

      if (!buffer) {
        llvm::consumeError(buffer.takeError());
        return false;
      }

      ast::Parser parser(input.file(), (*buffer)->getBufferStart());
      auto cls = parser.classDefinition();

      if (!cls) {
        llvm::consumeError(cls.takeError());
        return false;
      }

      instance().classes().push_back(std::move(*cls));
    }

    return true;
  }

  bool FrontendAction::runFrontendPasses()
  {
    marco::ast::PassManager frontendPassManager;
    frontendPassManager.addPass(ast::createTypeCheckingPass());
    frontendPassManager.addPass(ast::createConstantFolderPass());
    auto error = frontendPassManager.run(instance().classes());

    if (error) {
      llvm::consumeError(std::move(error));
      return false;
    }

    return true;
  }

  bool FrontendAction::runASTConversion()
  {
    marco::codegen::MLIRLowerer lowerer(instance().mlirContext());
    auto module = lowerer.run(instance().classes());

    if (!module) {
      return false;
    }

    instance().setMlirModule(std::make_unique<mlir::ModuleOp>(std::move(*module)));
    return true;
  }

  bool FrontendAction::runDialectConversion()
  {
    mlir::PassManager passManager(&instance().mlirContext());

    passManager.addPass(codegen::createAutomaticDifferentiationPass());
    //passManager.addPass(codegen::createSolveModelPass());
    passManager.addPass(codegen::createFunctionsVectorizationPass());
    passManager.addPass(codegen::createExplicitCastInsertionPass());

    passManager.addPass(codegen::createResultBuffersToArgsPass());

    passManager.addPass(mlir::createCanonicalizerPass());

    passManager.addPass(codegen::createFunctionConversionPass());

    // The buffer deallocation pass must be placed after the Modelica's
    // functions and members conversion, so that we can operate on an IR
    // without hidden allocs and frees.
    // However the pass must also be placed before the conversion of the
    // more common Modelica operations (i.e. add, sub, call, etc.), in
    // order to take into consideration their memory effects.
    passManager.addPass(codegen::createBufferDeallocationPass());

    passManager.addPass(codegen::createModelicaConversionPass());

    passManager.addPass(codegen::createLowerToCFGPass());
    passManager.addNestedPass<mlir::FuncOp>(mlir::createConvertMathToLLVMPass());
    passManager.addPass(codegen::createLLVMLoweringPass());

    return passManager.run(instance().mlirModule()).succeeded();
  }

  bool FrontendAction::runLLVMIRGeneration()
  {
    // Register the conversions to LLVM IR
    mlir::registerLLVMDialectTranslation(instance().mlirContext());
    mlir::registerOpenMPDialectTranslation(instance().mlirContext());

    // Initialize LLVM targets
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Convert to LLVM IR
    auto llvmModule = mlir::translateModuleToLLVMIR(instance().mlirModule(), instance().llvmContext());

    if (!llvmModule) {
      llvm::errs() << "Failed to emit LLVM IR\n";
      return false;
    }

    // Optimize the IR
    int optLevel = 2;

    auto optPipeline = mlir::makeOptimizingTransformer(optLevel, 0, nullptr);

    if (auto err = optPipeline(llvmModule.get())) {
      llvm::errs() << "Failed to optimize LLVM IR: " << err << "\n";
      return false;
    }

    instance().setLLVMModule(std::move(llvmModule));
    return true;
  }


  /*
  void FrontendAction::set_currentInput(const FrontendInputFile &currentInput) {
    this->currentInput_ = currentInput;
  }

  // Call this method if BeginSourceFile fails.
  // Deallocate compiler instance, input and output descriptors
  static void BeginSourceFileCleanUp(FrontendAction &fa, CompilerInstance &ci) {
    ci.ClearOutputFiles(true);
    fa.set_currentInput(FrontendInputFile());
    fa.set_instance(nullptr);
  }

  bool FrontendAction::BeginSourceFile(CompilerInstance& ci, const FrontendInputFile& realInput) {
    FrontendInputFile input(realInput);

    // Return immediately if the input file does not exist or is not a file.
    // Note that we cannot check this for input from stdin.

    if (input.file() != "-") {
      if (!llvm::sys::fs::is_regular_file(input.file())) {
        // Create a diagnostic ID to report
        unsigned int diagID;

        if (llvm::vfs::getRealFileSystem()->exists(input.file())) {
          ci.diagnostics().Report(clang::diag::err_fe_error_reading) << input.file();
          diagID = ci.diagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error, "%0 is not a regular file");
        } else {
          diagID = ci.diagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error, "%0 does not exist");
        }

        // Report the diagnostic and return
        ci.diagnostics().Report(diagID) << input.file();
        BeginSourceFileCleanUp(*this, ci);
        return false;
      }
    }

    assert(!instance_ && "Already processing a source file!");
    assert(!realInput.IsEmpty() && "Unexpected empty filename!");

    set_currentInput(realInput);
    set_instance(&ci);

    if (!ci.HasAllSources()) {
      BeginSourceFileCleanUp(*this, ci);
      return false;
    }

    if (!BeginSourceFileAction()) {
      BeginSourceFileCleanUp(*this, ci);
      return false;
    }

    return true;
  }

  bool FrontendAction::ShouldEraseOutputFiles() {
    return instance().diagnostics().hasErrorOccurred();
  }

  llvm::Error FrontendAction::Execute() {
    ExecuteAction();

    return llvm::Error::success();
  }

  void FrontendAction::EndSourceFile() {
    CompilerInstance &ci = instance();

    // Cleanup the output streams, and erase the output files if instructed by the
    // FrontendAction.
    ci.ClearOutputFiles(ShouldEraseOutputFiles());

    set_instance(nullptr);
    set_currentInput(FrontendInputFile());
  }

  bool FrontendAction::runParse() {
    CompilerInstance &ci = this->instance();

    ci.inputFiles()

    // Parse. In case of failure, report and return.
    ci.parsing().Parse(llvm::outs());

    if (reportFatalParsingErrors()) {
      return false;
    }

    // Report the diagnostics from parsing
    ci.parsing().messages().Emit(llvm::errs(), ci.allCookedSources());

    return true;
  }

  template <unsigned N>
  bool FrontendAction::reportFatalErrors(const char (&message)[N]) {

    if (!instance_->parsing().messages().empty() &&
        (instance_->invocation().warnAsErr() ||
            instance_->parsing().messages().AnyFatalError())) {
      const unsigned diagID = instance_->diagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, message);
      instance_->diagnostics().Report(diagID) << GetCurrentFileOrBufferName();
      instance_->parsing().messages().Emit(
          llvm::errs(), instance_->allCookedSources());
      return true;
    }
    return false;
  }
  */
}
