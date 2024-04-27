#ifndef MARCO_FRONTEND_COMPILERINSTANCE_H
#define MARCO_FRONTEND_COMPILERINSTANCE_H

#include "marco/AST/AST.h"
#include "marco/Frontend/CompilerInvocation.h"
#include "marco/Frontend/FrontendAction.h"
#include "marco/Frontend/SimulationOptions.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
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
      clang::DiagnosticsEngine& getDiagnostics();

      /// Get the current diagnostics engine.
      const clang::DiagnosticsEngine& getDiagnostics() const;

      clang::DiagnosticConsumer& getDiagnosticClient() const;

      /// @name File manager
      /// {

      bool hasFileManager() const;

      /// Get the file manager.
      clang::FileManager& getFileManager() const;

      /// Replace the current file manager.
      void setFileManager(clang::FileManager* value);

      /// @}
      /// @name Source Manager
      /// @{

      bool hasSourceManager() const;

      /// Get the source manager.
      clang::SourceManager& getSourceManager() const;

      /// Replace the current source manager.
      void setSourceManager(clang::SourceManager* value);

      /// }
      /// @name Forwarding methods
      /// {

      LanguageOptions& getLanguageOptions();
      const LanguageOptions& getLanguageOptions() const;

      clang::DiagnosticOptions& getDiagnosticOptions();
      const clang::DiagnosticOptions& getDiagnosticOptions() const;

      clang::FileSystemOptions& getFileSystemOpts();
      const clang::FileSystemOptions& getFileSystemOpts() const;

      FrontendOptions& getFrontendOptions();
      const FrontendOptions& getFrontendOptions() const;

      CodegenOptions& getCodeGenOptions();
      const CodegenOptions& getCodeGenOptions() const;

      SimulationOptions& getSimulationOptions();
      const SimulationOptions& getSimulationOptions() const;

      /// }

      /// Execute the provided action against the compiler's CompilerInvocation
      /// object.
      ///
      /// @param action the action to be executed
      /// @return whether the execution has been successful or not
      bool executeAction(FrontendAction& action);

      /// @name Construction Utility Methods
      /// @{

      /// Create the diagnostics engine using the invocation's diagnostic
      /// options and replace any existing one with it.
      ///
      /// Note that this routine also replaces the diagnostic client,
      /// allocating one if one is not provided.
      ///
      /// @param client If non-NULL, a diagnostic client that will be
      /// attached to (and, then, owned by) the DiagnosticsEngine.
      ///
      /// @param shouldOwnClient If client is non-NULL, specifies whether the
      /// diagnostic object should take ownership of the client.
      void createDiagnostics(
          clang::DiagnosticConsumer* client = nullptr,
          bool shouldOwnClient = true);

      /// Create a DiagnosticsEngine object with a the TextDiagnosticPrinter.
      ///
      /// If no diagnostic client is provided, this creates a
      /// DiagnosticConsumer that is owned by the returned diagnostic
      /// object, if using directly the caller is responsible for
      /// releasing the returned DiagnosticsEngine's client eventually.
      ///
      /// @param langaugeOptions - The language options
      /// @param diagnosticOptions - The diagnostic options
      ///
      /// @param client If non-NULL, a diagnostic client that will be
      /// attached to (and, then, owned by) the returned DiagnosticsEngine
      /// object.
      ///
      /// @param shouldOwnClient If client is non-NULL, specifies whether the
      /// diagnostic object should take ownership of the client.
      ///
      /// @return The new object on success, or null on failure.
      static llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine>
      createDiagnostics(
          LanguageOptions* languageOptions,
          clang::DiagnosticOptions* diagnosticOptions,
          clang::DiagnosticConsumer* client = nullptr,
          bool shouldOwnClient = true);

      /// Create the file manager and replace any existing one with it.
      ///
      /// @return the new file manager on success, or null on failure.
      clang::FileManager* createFileManager(
          llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS = nullptr);

      /// Create the source manager and replace any existing one with it.
      void createSourceManager(clang::FileManager& fileManager);

      /// }
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
      llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> diagnostics;

      /// The file manager.
      llvm::IntrusiveRefCntPtr<clang::FileManager> fileManager;

      /// The source manager.
      llvm::IntrusiveRefCntPtr<clang::SourceManager> sourceManager;

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
