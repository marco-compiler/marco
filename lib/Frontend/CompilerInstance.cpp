#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "marco/Frontend/CompilerInstance.h"
#include "marco/Frontend/CompilerInvocation.h"
#include "marco/Frontend/FrontendActions.h"
#include "marco/Frontend/Options.h"
#include "marco/Frontend/TextDiagnosticPrinter.h"
#include "marco/Codegen/dialects/modelica/ModelicaDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

// Helper method to generate the path of the output file. The following logic
// applies:
// 1. If the user specifies the output file via `-o`, then use that (i.e.
//    the outputFilename parameter).
// 2. If the user does not specify the name of the output file, derive it from
//    the input file (i.e. inputFilename + extension)
// 3. If the output file is not specified and the input file is `-`, then set
//    the output file to `-` as well.
static std::string getOutputFilePath(
    llvm::StringRef outputFilename, llvm::StringRef inputFilename, llvm::StringRef extension)
{
  // Output filename is specified. Just use that.
  if (!outputFilename.empty()) {
    return std::string(outputFilename);
  }

  // Output filename _is not_ specified. Derive it from the input file name.
  std::string outFile = "-";

  if (!extension.empty() && (inputFilename != "-")) {
    llvm::SmallString<128> path(inputFilename);
    llvm::sys::path::replace_extension(path, extension);
    outFile = std::string(path.str());
  }

  return outFile;
}

namespace marco::frontend
{
  CompilerInstance::CompilerInstance()
      : invocation_(new CompilerInvocation()),
        mlirContext_(std::make_unique<mlir::MLIRContext>()),
        llvmContext_(std::make_unique<llvm::LLVMContext>())
  {
    mlirContext_->loadDialect<marco::codegen::modelica::ModelicaDialect>();
    mlirContext_->loadDialect<mlir::StandardOpsDialect>();
  }

  CompilerInstance::~CompilerInstance()
  {
    assert(outputFiles_.empty() && "Still output files in flight?");
  }

  CompilerInvocation& CompilerInstance::invocation()
  {
    assert(invocation_ && "Compiler instance has no invocation!");
    return *invocation_;
  }

  void CompilerInstance::createDiagnostics(clang::DiagnosticConsumer* client, bool shouldOwnClient)
  {
    diagnostics_ = createDiagnostics(&getDiagnosticOptions(), client, shouldOwnClient);
  }

  clang::IntrusiveRefCntPtr<clang::DiagnosticsEngine> CompilerInstance::createDiagnostics(
      clang::DiagnosticOptions* options, clang::DiagnosticConsumer* client, bool shouldOwnClient)
  {
    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagID(new clang::DiagnosticIDs());
    clang::IntrusiveRefCntPtr<clang::DiagnosticsEngine> diags(new clang::DiagnosticsEngine(diagID, options));

    // Create the diagnostic client for reporting errors or for implementing -verify
    if (client) {
      diags->setClient(client, shouldOwnClient);
    } else {
      diags->setClient(new TextDiagnosticPrinter(llvm::errs(), options));
    }

    return diags;
  }

  clang::DiagnosticsEngine& CompilerInstance::getDiagnostics() const
  {
    assert(diagnostics_ && "Compiler instance has no diagnostics!");
    return *diagnostics_;
  }

  clang::DiagnosticConsumer& CompilerInstance::getDiagnosticClient() const
  {
    assert(diagnostics_ && diagnostics_->getClient() &&
        "Compiler instance has no diagnostic client!");
    return *diagnostics_->getClient();
  }

  FrontendOptions& CompilerInstance::getFrontendOptions()
  {
    return invocation_->frontendOptions();
  }

  const FrontendOptions& CompilerInstance::getFrontendOptions() const
  {
    return invocation_->frontendOptions();
  }

  CodegenOptions& CompilerInstance::getCodegenOptions()
  {
    return invocation_->codegenOptions();
  }

  const CodegenOptions& CompilerInstance::getCodegenOptions() const
  {
    return invocation_->codegenOptions();
  }

  clang::DiagnosticOptions& CompilerInstance::getDiagnosticOptions()
  {
    return invocation_->GetDiagnosticOpts();
  }

  const SimulationOptions& CompilerInstance::getSimulationOptions() const
  {
    return invocation_->simulationOptions();
  }

  SimulationOptions& CompilerInstance::getSimulationOptions()
  {
    return invocation_->simulationOptions();
  }

  const clang::DiagnosticOptions& CompilerInstance::getDiagnosticOptions() const
  {
    return invocation_->GetDiagnosticOpts();
  }

  bool CompilerInstance::executeAction(FrontendAction& action)
  {
    action.setCompilerInstance(this);

    // Run the frontend action
    if (action.beginAction()) {
      action.execute();
    }

    return getDiagnostics().getClient()->getNumErrors() == 0;
  }

  void CompilerInstance::clearOutputFiles(bool eraseFiles)
  {
    for (OutputFile& of: outputFiles_) {
      if (!of.fileName.empty() && eraseFiles) {
        llvm::sys::fs::remove(of.fileName);
      }
    }

    outputFiles_.clear();
  }

  mlir::MLIRContext& CompilerInstance::getMLIRContext()
  {
    assert(mlirContext_ != nullptr && "MLIR context not set");
    return *mlirContext_;
  }

  const mlir::MLIRContext& CompilerInstance::getMLIRContext() const
  {
    assert(mlirContext_ != nullptr && "MLIR context not set");
    return *mlirContext_;
  }

  llvm::LLVMContext& CompilerInstance::getLLVMContext()
  {
    assert(llvmContext_ != nullptr && "LLVM context not set");
    return *llvmContext_;
  }

  const llvm::LLVMContext& CompilerInstance::getLLVMContext() const
  {
    assert(llvmContext_ != nullptr && "LLVM context not set");
    return *llvmContext_;
  }

  std::string& CompilerInstance::getFlattened()
  {
    return flattened_;
  }

  const std::string& CompilerInstance::getFlattened() const
  {
    return flattened_;
  }

  void CompilerInstance::setFlattened(std::string value)
  {
    flattened_ = value;
  }

  std::unique_ptr<ast::Class>& CompilerInstance::getAST()
  {
    return ast_;
  }

  const std::unique_ptr<ast::Class>& CompilerInstance::getAST() const
  {
    return ast_;
  }

  void CompilerInstance::setAST(std::unique_ptr<ast::Class> value)
  {
    ast_ = std::move(value);
  }

  mlir::ModuleOp& CompilerInstance::getMLIRModule()
  {
    assert(mlirModule_ != nullptr && "MLIR module not set");
    return *mlirModule_;
  }

  const mlir::ModuleOp& CompilerInstance::getMLIRModule() const
  {
    assert(mlirModule_ != nullptr && "MLIR module not set");
    return *mlirModule_;
  }

  void CompilerInstance::setMLIRModule(std::unique_ptr<mlir::ModuleOp> module)
  {
    mlirModule_ = std::move(module);
  }

  llvm::Module& CompilerInstance::getLLVMModule()
  {
    assert(mlirModule_ != nullptr && "LLVM module not set");
    return *llvmModule_;
  }

  const llvm::Module& CompilerInstance::getLLVMModule() const
  {
    assert(mlirModule_ != nullptr && "LLVM module not set");
    return *llvmModule_;
  }

  void CompilerInstance::setLLVMModule(std::unique_ptr<llvm::Module> module)
  {
    llvmModule_ = std::move(module);
  }

  std::unique_ptr<llvm::raw_pwrite_stream> CompilerInstance::createDefaultOutputFile(
      bool binary, llvm::StringRef baseName, llvm::StringRef extension)
  {
    // Get the path of the output file
    std::string outputFilePath = getOutputFilePath(getFrontendOptions().outputFile, baseName, extension);

    // Create the output file
    llvm::Expected<std::unique_ptr<llvm::raw_pwrite_stream>> os = createOutputFileImpl(outputFilePath, binary);

    // If successful, add the file to the list of tracked output files and return
    if (os) {
      outputFiles_.emplace_back(OutputFile(outputFilePath));
      return std::move(*os);
    }

    // If unsuccessful, issue an error and return null
    unsigned DiagID = getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "unable to open output file '%0': '%1'");

    getDiagnostics().Report(DiagID) << outputFilePath << llvm::errorToErrorCode(os.takeError()).message();
    return nullptr;
  }

  llvm::Expected<std::unique_ptr<llvm::raw_pwrite_stream>> CompilerInstance::createOutputFileImpl(
      llvm::StringRef outputFilePath, bool binary)
  {
    // Creates the file descriptor for the output file
    std::unique_ptr<llvm::raw_fd_ostream> os;

    std::error_code error;

    os.reset(
        new llvm::raw_fd_ostream(
            outputFilePath, error,
            (binary ? llvm::sys::fs::OF_None : llvm::sys::fs::OF_TextWithCRLF)));

    if (error) {
      return llvm::errorCodeToError(error);
    }

    // For seekable streams, just return the stream corresponding to the output file
    if (!binary || os->supportsSeeking()) {
      return std::move(os);
    }

    // For non-seekable streams, we need to wrap the output stream into something
    // that supports 'pwrite' and takes care of the ownership for us.
    return std::make_unique<llvm::buffer_unique_ostream>(std::move(os));
  }
}
