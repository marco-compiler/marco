#include "marco/Frontend/CompilerInstance.h"
#include "marco/Diagnostic/Printer.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Frontend/CompilerInvocation.h"
#include "marco/Frontend/FrontendActions.h"
#include "clang/Driver/Options.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco;
using namespace ::marco::diagnostic;
using namespace ::marco::frontend;
using namespace ::marco::io;

//===---------------------------------------------------------------------===//
// Messages
//===---------------------------------------------------------------------===//

namespace
{
  class CantOpenFileMessage : public Message
  {
    public:
      CantOpenFileMessage(llvm::StringRef file, llvm::StringRef error)
        : file(file.str()),
          error(error.str())
      {
      }

      void print(PrinterInstance* printer) const override
      {
        auto& os = printer->getOutputStream();
        os << "Unable to open file '" << file << "': " << error << "\n";
      }

    private:
      std::string file;
      std::string error;
  };
}

//===---------------------------------------------------------------------===//
// CompilerInstance
//===---------------------------------------------------------------------===//

namespace marco::frontend
{
  CompilerInstance::OutputFile::OutputFile(
      llvm::StringRef fileName,  std::optional<llvm::sys::fs::TempFile> file)
      : fileName(fileName.str()), file(std::move(file))
  {
  }

  CompilerInstance::CompilerInstance()
      : invocation(new CompilerInvocation()),
        diagnostics(
            std::make_unique<DiagnosticEngine>(std::make_unique<Printer>()))
  {
  }

  CompilerInstance::~CompilerInstance()
  {
    assert(outputFiles.empty() && "Still output files in flight?");
  }

  bool CompilerInstance::hasInvocation() const
  {
    return invocation != nullptr;
  }

  CompilerInvocation& CompilerInstance::getInvocation()
  {
    assert(invocation && "Compiler instance has no invocation");
    return *invocation;
  }

  const CompilerInvocation& CompilerInstance::getInvocation() const
  {
    assert(invocation && "Compiler instance has no invocation");
    return *invocation;
  }

  void CompilerInstance::setInvocation(
      std::shared_ptr<CompilerInvocation> value)
  {
    invocation = value;
  }

  bool CompilerInstance::hasDiagnostics() const
  {
    return diagnostics != nullptr;
  }

  diagnostic::DiagnosticEngine& CompilerInstance::getDiagnostics() const
  {
    assert(diagnostics && "Compiler instance has no diagnostics");
    return *diagnostics;
  }

  FrontendOptions& CompilerInstance::getFrontendOptions()
  {
    return getInvocation().getFrontendOptions();
  }

  const FrontendOptions& CompilerInstance::getFrontendOptions() const
  {
    return getInvocation().getFrontendOptions();
  }

  CodegenOptions& CompilerInstance::getCodeGenOptions()
  {
    return getInvocation().getCodeGenOptions();
  }

  const CodegenOptions& CompilerInstance::getCodeGenOptions() const
  {
    return getInvocation().getCodeGenOptions();
  }

  SimulationOptions& CompilerInstance::getSimulationOptions()
  {
    return getInvocation().getSimulationOptions();
  }

  const SimulationOptions& CompilerInstance::getSimulationOptions() const
  {
    return getInvocation().getSimulationOptions();
  }

  diagnostic::DiagnosticOptions& CompilerInstance::getDiagnosticOptions()
  {
    return getDiagnostics().getOptions();
  }

  const diagnostic::DiagnosticOptions&
  CompilerInstance::getDiagnosticOptions() const
  {
    return getDiagnostics().getOptions();
  }

  bool CompilerInstance::executeAction(FrontendAction& act)
  {
    assert(hasDiagnostics() && "Diagnostics engine is not initialized");
    assert(!getFrontendOptions().showHelp && "Client must handle '--help'");

    assert(!getFrontendOptions().showVersion &&
           "Client must handle '--version'");

    if (!act.prepareToExecute(*this)) {
      return false;
    }

    if (act.beginSourceFiles(*this, getFrontendOptions().inputs)) {
      if (llvm::Error err = act.execute()) {
        consumeError(std::move(err));
      }

      act.endSourceFiles();
    }

    return getDiagnostics().numOfErrors() == 0;
  }

  void CompilerInstance::clearOutputFiles(bool eraseFiles)
  {
    // The AST consumer can own streams that write to the output files.
    //assert(!hasASTConsumer() && "AST consumer should be reset");

    // Ignore errors that occur when trying to discard the temporary file.
    for (OutputFile& of: outputFiles) {
      if (eraseFiles) {
        if (of.file) {
          llvm::consumeError(of.file->discard());
        }

        if (!of.fileName.empty()) {
          llvm::sys::fs::remove(of.fileName);
        }

        continue;
      }

      if (!of.file) {
        continue;
      }

      if (of.file->TmpName.empty()) {
        llvm::consumeError(of.file->discard());
        continue;
      }

      llvm::Error e = of.file->keep(of.fileName);

      if (!e) {
        continue;
      }

      // TODO print error of file renaming

      llvm::sys::fs::remove(of.file->TmpName);
    }

    outputFiles.clear();
  }

  std::unique_ptr<llvm::raw_pwrite_stream>
  CompilerInstance::createDefaultOutputFile(
      bool binary,
      llvm::StringRef inFile,
      llvm::StringRef extension,
      bool createMissingDirectories,
      bool forceUseTemporary)
  {
    llvm::StringRef outputPath = getFrontendOptions().outputFile;

    if (outputPath.empty()) {
      if (inFile == "-" || extension.empty()) {
        outputPath = "-";
      } else {
        std::optional<llvm::SmallString<128>> pathStorage;
        pathStorage.emplace(inFile);
        llvm::sys::path::replace_extension(*pathStorage, extension);
        outputPath = *pathStorage;
      }
    }

    // TODO check useTemporary from FrontendOptions
    return createOutputFile(
        outputPath, binary, forceUseTemporary, createMissingDirectories);
  }

  std::unique_ptr<llvm::raw_pwrite_stream> CompilerInstance::createOutputFile(
      llvm::StringRef outputPath,
      bool binary,
      bool useTemporary,
      bool createMissingDirectories)
  {
    llvm::Expected<std::unique_ptr<llvm::raw_pwrite_stream>> os =
        createOutputFileImpl(
            outputPath, binary, useTemporary, createMissingDirectories);

    if (os) {
      return std::move(*os);
    }

    getDiagnostics().emitError<CantOpenFileMessage>(
        outputPath,
        llvm::errorToErrorCode(os.takeError()).message());

    return nullptr;
  }

  llvm::Expected<std::unique_ptr<llvm::raw_pwrite_stream>>
  CompilerInstance::createOutputFileImpl(
      llvm::StringRef outputPath,
      bool binary,
      bool useTemporary,
      bool createMissingDirectories)
  {
    assert(!createMissingDirectories || useTemporary);

    std::unique_ptr<llvm::raw_fd_ostream> os;
    std::optional<llvm::StringRef> osFile;

    if (useTemporary) {
      if (outputPath == "-") {
        useTemporary = false;
      } else {
        llvm::sys::fs::file_status status;
        llvm::sys::fs::status(outputPath, status);

        if (llvm::sys::fs::exists(status)) {
          // Fail early if we can't write to the final destination.
          if (!llvm::sys::fs::can_write(outputPath)) {
            return llvm::errorCodeToError(
                llvm::make_error_code(llvm::errc::operation_not_permitted));
          }

          // Don't use a temporary if the output is a special file.
          // This handles things like '-o /dev/null'.
          if (!llvm::sys::fs::is_regular_file(status)) {
            useTemporary = false;
          }
        }
      }
    }

    std::optional<llvm::sys::fs::TempFile> temp;

    if (useTemporary) {
      llvm::StringRef outputExtension = llvm::sys::path::extension(outputPath);

      llvm::SmallString<128> tempPath =
          llvm::StringRef(outputPath).drop_back(outputExtension.size());

      tempPath += "-%%%%%%%%";
      tempPath += outputExtension;
      tempPath += ".tmp";

      llvm::Expected<llvm::sys::fs::TempFile> expectedFile =
          llvm::sys::fs::TempFile::create(
              tempPath, llvm::sys::fs::all_read | llvm::sys::fs::all_write,
              binary ? llvm::sys::fs::OF_None : llvm::sys::fs::OF_Text);

      llvm::Error e = handleErrors(
          expectedFile.takeError(), [&](const llvm::ECError &E) -> llvm::Error {
            std::error_code ec = E.convertToErrorCode();

            if (createMissingDirectories &&
                ec == llvm::errc::no_such_file_or_directory) {
              llvm::StringRef Parent =
                  llvm::sys::path::parent_path(outputPath);

              ec = llvm::sys::fs::create_directories(Parent);

              if (!ec) {
                expectedFile = llvm::sys::fs::TempFile::create(tempPath);

                if (!expectedFile) {
                  return llvm::errorCodeToError(
                      llvm::errc::no_such_file_or_directory);
                }
              }
            }

            return llvm::errorCodeToError(ec);
          });

      if (e) {
        consumeError(std::move(e));
      } else {
        temp = std::move(expectedFile.get());
        os.reset(new llvm::raw_fd_ostream(temp->FD, false));
        osFile = temp->TmpName;
      }
    }

    if (!os) {
      osFile = outputPath;
      std::error_code ec;

      os.reset(new llvm::raw_fd_ostream(
          *osFile, ec,
          (binary ? llvm::sys::fs::OF_None : llvm::sys::fs::OF_TextWithCRLF)));

      if (ec) {
        return llvm::errorCodeToError(ec);
      }
    }

    // Add the output file -- but don't try to remove "-", since this means we
    // are using stdin.
    outputFiles.emplace_back(
        ((outputPath != "-") ? outputPath : "").str(), std::move(temp));

    if (!binary || os->supportsSeeking()) {
      return std::move(os);
    }

    return std::make_unique<llvm::buffer_unique_ostream>(std::move(os));
  }
}
