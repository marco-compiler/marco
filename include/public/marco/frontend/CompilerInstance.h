#ifndef MARCO_FRONTEND_COMPILERINSTANCE_H
#define MARCO_FRONTEND_COMPILERINSTANCE_H

#include <clang/Basic/DiagnosticOptions.h>
#include <list>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <marco/ast/AST.h>

#include "CompilerInvocation.h"
#include "FrontendAction.h"

namespace marco::frontend
{
  /// Helper class for managing a single instance of the Flang compiler.
  ///
  /// This class serves two purposes:
  ///  (1) It manages the various objects which are necessary to run the compiler
  ///  (2) It provides utility routines for constructing and manipulating the
  ///      common Flang objects.
  ///
  /// The compiler instance generally owns the instance of all the objects that it
  /// manages. However, clients can still share objects by manually setting the
  /// object and retaking ownership prior to destroying the CompilerInstance.
  ///
  /// The compiler instance is intended to simplify clients, but not to lock them
  /// in to the compiler instance for everything. When possible, utility functions
  /// come in two forms; a short form that reuses the CompilerInstance objects,
  /// and a long form that takes explicit instances of any required objects.
  class CompilerInstance
  {
      /// The options used in this compiler instance.
      std::shared_ptr<CompilerInvocation> invocation_;

      /// The diagnostics engine instance.
      llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> diagnostics_;

      /// Holds information about the output file.
      struct OutputFile
      {
        std::string filename_;

        OutputFile(std::string inputFilename)
            : filename_(std::move(inputFilename)) {}
      };

      /// The list of active output files.
      std::list<OutputFile> outputFiles_;

      /// Holds the output stream provided by the user. Normally, users of
      /// CompilerInstance will call CreateOutputFile to obtain/create an output
      /// stream. If they want to provide their own output stream, this field will
      /// facilitate this. It is optional and will normally be just a nullptr.
      std::unique_ptr<llvm::raw_pwrite_stream> outputStream_;

    public:
      explicit CompilerInstance();

      ~CompilerInstance();

      CompilerInvocation& invocation()
      {
        assert(invocation_ && "Compiler instance has no invocation!");
        return *invocation_;
      };

      /// Replace the current invocation.
      void set_invocation(std::shared_ptr<CompilerInvocation> value);

      /// Return the current allSources.
      /*
      Fortran::parser::AllSources& allSources() const { return *allSources_; }

      bool HasAllSources() const { return allSources_ != nullptr; }

      parser::AllCookedSources& allCookedSources()
      {
        assert(allCookedSources_ && "Compiler instance has no AllCookedSources!");
        return *allCookedSources_;
      };

      /// }
      /// @name Parser Operations
      /// {

      /// Return parsing to be used by Actions.
      Fortran::parser::Parsing& parsing() const { return *parsing_; }
       */

      bool ExecuteAction(FrontendAction& act);

      clang::DiagnosticOptions& GetDiagnosticOpts()
      {
        return invocation_->GetDiagnosticOpts();
      }

      const clang::DiagnosticOptions& GetDiagnosticOpts() const
      {
        return invocation_->GetDiagnosticOpts();
      }

      FrontendOptions& frontendOpts() { return invocation_->frontendOptions(); }

      const FrontendOptions& frontendOpts() const
      {
        return invocation_->frontendOptions();
      }

      /// Get the current diagnostics engine.
      clang::DiagnosticsEngine& diagnostics() const
      {
        assert(diagnostics_ && "Compiler instance has no diagnostics!");
        return *diagnostics_;
      }

      clang::DiagnosticConsumer& GetDiagnosticClient() const
      {
        assert(diagnostics_ && diagnostics_->getClient() &&
            "Compiler instance has no diagnostic client!");
        return *diagnostics_->getClient();
      }

      /// Clear the output file list.
      void ClearOutputFiles(bool eraseFiles);

      /// Create the default output file (based on the invocation's options) and
      /// add it to the list of tracked output files. If the name of the output
      /// file is not provided, it will be derived from the input file.
      ///
      /// \param binary     The mode to open the file in.
      /// \param baseInput  If the invocation contains no output file name (i.e.
      ///                   outputFile in FrontendOptions is empty), the input path
      ///                   name to use for deriving the output path.
      /// \param extension  The extension to use for output names derived from
      ///                   \p baseInput.
      /// \return           Null on error, ostream for the output file otherwise
      std::unique_ptr<llvm::raw_pwrite_stream> CreateDefaultOutputFile(
          bool binary = true, llvm::StringRef baseInput = "",
          llvm::StringRef extension = "");

    private:
      /// Create a new output file
      ///
      /// \param outputPath   The path to the output file.
      /// \param binary       The mode to open the file in.
      /// \return             Null on error, ostream for the output file otherwise
      llvm::Expected<std::unique_ptr<llvm::raw_pwrite_stream>> CreateOutputFileImpl(
          llvm::StringRef outputPath, bool binary);

    public:
      /// Create a DiagnosticsEngine object
      ///
      /// If no diagnostic client is provided, this method creates a
      /// DiagnosticConsumer that is owned by the returned diagnostic object. If
      /// using directly the caller is responsible for releasing the returned
      /// DiagnosticsEngine's client eventually.
      ///
      /// \param opts - The diagnostic options; note that the created text
      /// diagnostic object contains a reference to these options.
      ///
      /// \param client - If non-NULL, a diagnostic client that will be attached to
      /// (and optionally, depending on /p shouldOwnClient, owned by) the returned
      /// DiagnosticsEngine object.
      ///
      /// \return The new object on success, or null on failure.
      static clang::IntrusiveRefCntPtr<clang::DiagnosticsEngine> CreateDiagnostics(
          clang::DiagnosticOptions* opts,
          clang::DiagnosticConsumer* client = nullptr,
          bool shouldOwnClient = true);

      void CreateDiagnostics(clang::DiagnosticConsumer* client = nullptr, bool shouldOwnClient = true);

      void set_outputStream(std::unique_ptr<llvm::raw_pwrite_stream> outStream)
      {
        outputStream_ = std::move(outStream);
      }

      bool IsOutputStreamNull() { return (outputStream_ == nullptr); }

      // Allow the frontend compiler to write in the output stream.
      void WriteOutputStream(const std::string& message)
      {
        *outputStream_ << message;
      }

    public:
      std::vector<std::unique_ptr<ast::Class>>& classes()
      {
        return classes_;
      }

      const std::vector<std::unique_ptr<ast::Class>>& classes() const
      {
        return classes_;
      }

      mlir::MLIRContext& mlirContext()
      {
        assert(mlirContext_ != nullptr && "MLIR context not set");
        return *mlirContext_;
      }

      const mlir::MLIRContext& mlirContext() const
      {
        assert(mlirContext_ != nullptr && "MLIR context not set");
        return *mlirContext_;
      }

      mlir::ModuleOp& mlirModule()
      {
        assert(mlirModule_ != nullptr && "MLIR module not set");
        return *mlirModule_;
      }

      const mlir::ModuleOp& mlirModule() const
      {
        assert(mlirModule_ != nullptr && "MLIR module not set");
        return *mlirModule_;
      }

      void setMlirModule(std::unique_ptr<mlir::ModuleOp> module)
      {
        mlirModule_ = std::move(module);
      }

      llvm::LLVMContext& llvmContext()
      {
        assert(llvmContext_ != nullptr && "LLVM context not set");
        return *llvmContext_;
      }

      const llvm::LLVMContext& llvmContext() const
      {
        assert(llvmContext_ != nullptr && "LLVM context not set");
        return *llvmContext_;
      }

      llvm::Module& llvmModule()
      {
        assert(mlirModule_ != nullptr && "LLVM module not set");
        return *llvmModule_;
      }

      const llvm::Module& llvmModule() const
      {
        assert(mlirModule_ != nullptr && "LLVM module not set");
        return *llvmModule_;
      }

      void setLLVMModule(std::unique_ptr<llvm::Module> module)
      {
        llvmModule_ = std::move(module);
      }

    private:
      std::vector<std::unique_ptr<ast::Class>> classes_;
      std::unique_ptr<mlir::MLIRContext> mlirContext_;
      std::unique_ptr<mlir::ModuleOp> mlirModule_;
      std::unique_ptr<llvm::LLVMContext> llvmContext_;
      std::unique_ptr<llvm::Module> llvmModule_;
  };
}

#endif // MARCO_FRONTEND_COMPILERINSTANCE_H
