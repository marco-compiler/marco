#ifndef MARCO_FRONTEND_COMPILERINSTANCE_H
#define MARCO_FRONTEND_COMPILERINSTANCE_H

#include "marco/AST/AST.h"
#include "marco/Diagnostic/Diagnostic.h"
#include "marco/Frontend/CompilerInvocation.h"
#include "marco/Frontend/FrontendAction.h"
#include "marco/Frontend/SimulationOptions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <list>

namespace marco::frontend
{
  /// Helper class for managing a single instance of the MARCO compiler.
  class CompilerInstance
  {
    public:
      /// Holds information about the output file.
      struct OutputFile
      {
        std::string fileName;
        std::optional<llvm::sys::fs::TempFile> file;

        OutputFile(
            llvm::StringRef fileName,
            std::optional<llvm::sys::fs::TempFile> file);
      };

      CompilerInstance();

      ~CompilerInstance();

      CompilerInstance(const CompilerInstance&) = delete;

      void operator=(const CompilerInstance&) = delete;

      /// @name Compiler invocation
      /// {

      /// Check whether an invocation has been set for this compiler instance.
      bool hasInvocation() const;

      /// Get the current compiler invocation.
      CompilerInvocation& getInvocation();

      /// Get the current compiler invocation.
      const CompilerInvocation& getInvocation() const;

      /// Replace the current invocation.
      void setInvocation(std::shared_ptr<CompilerInvocation> value);

      /// }
      /// @name Diagnostics
      /// {

      /// CHeck whether the diagnostic engine has been set.
      bool hasDiagnostics() const;

      /// Get the current diagnostics engine.
      diagnostic::DiagnosticEngine& getDiagnostics() const;

      /// }
      /// @name Forwarding methods
      /// {

      FrontendOptions& getFrontendOptions();
      const FrontendOptions& getFrontendOptions() const;

      CodegenOptions& getCodeGenOptions();
      const CodegenOptions& getCodeGenOptions() const;

      SimulationOptions& getSimulationOptions();
      const SimulationOptions& getSimulationOptions() const;

      diagnostic::DiagnosticOptions& getDiagnosticOptions();
      const diagnostic::DiagnosticOptions& getDiagnosticOptions() const;

      /// }

      /// Execute the provided action against the compiler's CompilerInvocation
      /// object.
      ///
      /// @param action the action to be executed
      /// @return whether the execution has been successful or not
      bool executeAction(FrontendAction& action);

      /// @name Output Files
      /// {

      /// Clear the output file list.
      /// The underlying output streams must have been closed beforehand.
      void clearOutputFiles(bool eraseFiles);

      /// Create the default output file (based on the invocation's options) and
      /// add it to the list of tracked output files.
      std::unique_ptr<llvm::raw_pwrite_stream> createDefaultOutputFile(
          bool binary = true,
          llvm::StringRef inFile = "",
          llvm::StringRef extension = "",
          bool createMissingDirectories = false,
          bool forceUseTemporary = false);

      /// Create a new output file, optionally deriving the output path name,
      /// and add it to the list of tracked output files.
      std::unique_ptr<llvm::raw_pwrite_stream> createOutputFile(
          llvm::StringRef outputPath,
          bool binary,
          bool useTemporary = false,
          bool createMissingDirectories = false);

    private:
      /// Create a new output file.
      ///
      /// @param outputPath the path to the output file.
      /// @param binary the mode to open the file in.
      /// @param useTemporary create a new temporary file that must be renamed
      /// to outputPath in the end.
      /// @param createMissingDirectories whether to create the missing
      /// directories in the output path.
      llvm::Expected<std::unique_ptr<llvm::raw_pwrite_stream>>
      createOutputFileImpl(
        llvm::StringRef outputPath,
        bool binary,
        bool useTemporary,
        bool createMissingDirectories);

      /// }
      /// @name Output stream methods
      /// {

    public:
      void setOutputStream(std::unique_ptr<llvm::raw_pwrite_stream> outStream)
      {
        outputStream = std::move(outStream);
      }

      bool isOutputStreamNull()
      {
        return (outputStream == nullptr);
      }

      // Allow the frontend compiler to write in the output stream.
      void writeOutputStream(const std::string &message)
      {
        *outputStream << message;
      }

      /// Get the user specified output stream.
      llvm::raw_pwrite_stream& getOutputStream()
      {
        assert(outputStream &&
               "Compiler instance has no user-specified output stream");
        return *outputStream;
      }

      /// }

    private:
      /// The options used in this compiler instance.
      std::shared_ptr<CompilerInvocation> invocation;

      /// The diagnostics engine instance.
      std::unique_ptr<diagnostic::DiagnosticEngine> diagnostics;

      /// The list of active output files.
      std::list<OutputFile> outputFiles;

      /// Holds the output stream provided by the user. Normally, users of
      /// CompilerInstance will call createOutputFile to obtain/create an output
      /// stream. If they want to provide their own output stream, this field
      /// will facilitate this. It is optional and will normally be just a
      /// nullptr.
      std::unique_ptr<llvm::raw_pwrite_stream> outputStream;
  };

  /// Construct the FrontendAction of a compiler invocation based on the
  /// options specified for the compiler invocation.
  std::unique_ptr<FrontendAction> createFrontendAction(CompilerInstance& ci);

  /// Execute the given actions described by the compiler invocation object
  /// in the given compiler instance.
  ///
  /// @return true on success; false otherwise
  bool executeCompilerInvocation(CompilerInstance* instance);
}

#endif // MARCO_FRONTEND_COMPILERINSTANCE_H
