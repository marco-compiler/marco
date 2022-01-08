#include <llvm/Support/Errc.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>
#include <marco/frontend/CompilerInstance.h>
#include <marco/frontend/CompilerInvocation.h>
#include <marco/frontend/TextDiagnosticPrinter.h>
#include <marco/codegen/dialects/modelica/ModelicaDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

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

  CompilerInstance::~CompilerInstance() {
    assert(outputFiles_.empty() && "Still output files in flight?");
  }

  void CompilerInstance::set_invocation(
      std::shared_ptr<CompilerInvocation> value) {
    invocation_ = std::move(value);
  }

  // Helper method to generate the path of the output file. The following logic
  // applies:
  // 1. If the user specifies the output file via `-o`, then use that (i.e.
  //    the outputFilename parameter).
  // 2. If the user does not specify the name of the output file, derive it from
  //    the input file (i.e. inputFilename + extension)
  // 3. If the output file is not specified and the input file is `-`, then set
  //    the output file to `-` as well.
  static std::string GetOutputFilePath(llvm::StringRef outputFilename,
                                       llvm::StringRef inputFilename, llvm::StringRef extension) {

    // Output filename _is_ specified. Just use that.
    if (!outputFilename.empty())
      return std::string(outputFilename);

    // Output filename _is not_ specified. Derive it from the input file name.
    std::string outFile = "-";
    if (!extension.empty() && (inputFilename != "-")) {
      llvm::SmallString<128> path(inputFilename);
      llvm::sys::path::replace_extension(path, extension);
      outFile = std::string(path.str());
    }

    return outFile;
  }

  std::unique_ptr<llvm::raw_pwrite_stream> CompilerInstance::CreateDefaultOutputFile(
      bool binary, llvm::StringRef baseName, llvm::StringRef extension)
  {
    // Get the path of the output file
    std::string outputFilePath =
        GetOutputFilePath(frontendOpts().outputFile, baseName, extension);

    // Create the output file
    llvm::Expected<std::unique_ptr<llvm::raw_pwrite_stream>> os =
        CreateOutputFileImpl(outputFilePath, binary);

    // If successful, add the file to the list of tracked output files and
    // return.
    if (os) {
      outputFiles_.emplace_back(OutputFile(outputFilePath));
      return std::move(*os);
    }

    // If unsuccessful, issue an error and return Null
    unsigned DiagID = diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "unable to open output file '%0': '%1'");
    diagnostics().Report(DiagID)
        << outputFilePath << llvm::errorToErrorCode(os.takeError()).message();

    return nullptr;
  }

  llvm::Expected<std::unique_ptr<llvm::raw_pwrite_stream>> CompilerInstance::CreateOutputFileImpl(
      llvm::StringRef outputFilePath, bool binary)
  {
    // Creates the file descriptor for the output file
    std::unique_ptr<llvm::raw_fd_ostream> os;

    std::error_code error;
    os.reset(new llvm::raw_fd_ostream(outputFilePath, error,
        (binary ? llvm::sys::fs::OF_None : llvm::sys::fs::OF_TextWithCRLF)));
    if (error) {
      return llvm::errorCodeToError(error);
    }

    // For seekable streams, just return the stream corresponding to the output
    // file.
    if (!binary || os->supportsSeeking())
      return std::move(os);

    // For non-seekable streams, we need to wrap the output stream into something
    // that supports 'pwrite' and takes care of the ownership for us.
    return std::make_unique<llvm::buffer_unique_ostream>(std::move(os));
  }

  void CompilerInstance::ClearOutputFiles(bool eraseFiles) {
    for (OutputFile &of : outputFiles_)
      if (!of.filename_.empty() && eraseFiles)
        llvm::sys::fs::remove(of.filename_);

    outputFiles_.clear();
  }

  bool CompilerInstance::ExecuteAction(FrontendAction &act) {
    auto& invoc = this->invocation();

    // Set some sane defaults for the frontend.
    //invoc.SetDefaultFortranOpts();
    // Update the fortran options based on user-based input.
    //invoc.setFortranOpts();
    // Set the encoding to read all input files in based on user input.
    //allSources_->set_encoding(invoc.fortranOpts().encoding);
    // Create the semantics context and set semantic options.
    //invoc.setSemanticsOpts(*this->allCookedSources_);

    act.setCompilerInstance(*this);
    act.execute();

    /*
    // Run the frontend action `act` for every input file.ì
    for (const FrontendInputFile &fif : frontendOpts().inputs) {
      if (act.BeginSourceFile(*this, fif)) {
        if (llvm::Error err = act.Execute()) {
          consumeError(std::move(err));
        }

        act.EndSourceFile();
      }
    }
     */

    return diagnostics().getClient()->getNumErrors() == 0;
  }

  void CompilerInstance::CreateDiagnostics(
      clang::DiagnosticConsumer *client, bool shouldOwnClient) {
    diagnostics_ = CreateDiagnostics(&GetDiagnosticOpts(), client, shouldOwnClient);
  }

  clang::IntrusiveRefCntPtr<clang::DiagnosticsEngine>
  CompilerInstance::CreateDiagnostics(clang::DiagnosticOptions *opts,
                                      clang::DiagnosticConsumer *client, bool shouldOwnClient) {
    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagID(
        new clang::DiagnosticIDs());
    clang::IntrusiveRefCntPtr<clang::DiagnosticsEngine> diags(
        new clang::DiagnosticsEngine(diagID, opts));

    // Create the diagnostic client for reporting errors or for
    // implementing -verify.
    if (client) {
      diags->setClient(client, shouldOwnClient);
    } else {
      diags->setClient(new TextDiagnosticPrinter(llvm::errs(), opts));
    }

    return diags;
  }
}
