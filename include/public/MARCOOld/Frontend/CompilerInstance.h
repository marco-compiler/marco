#ifndef MARCO_FRONTEND_COMPILERINSTANCE_H
#define MARCO_FRONTEND_COMPILERINSTANCE_H

#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "marco/AST/AST.h"
#include "marco/Frontend/CompilerInvocation.h"
#include "marco/Frontend/FrontendAction.h"
#include "marco/Frontend/SimulationOptions.h"
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

        OutputFile(std::string fileName)
            : fileName(std::move(fileName))
        {
        }
      };

      explicit CompilerInstance();

      ~CompilerInstance();

    public:
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
      CompilerInstance(const CompilerInstance&) = delete;

      void operator=(const CompilerInstance&) = delete;

      // TODO rename to getInvocation()
      /// Get the current compiler invocation.
      CompilerInvocation& invocation();

      // TODO
      /// setInvocation - Replace the current invocation.
      //void setInvocation(std::shared_ptr<CompilerInvocation> Value);

      /// Create a diagnostics engine instance
      ///
      /// If no diagnostic client is provided, this method creates a
      /// DiagnosticConsumer that is owned by the returned diagnostic object. If
      /// using directly, the caller is responsible for releasing the returned
      /// DiagnosticsEngine's client eventually.
      ///
      /// @param options  diagnostic options
      /// @param client - If non-NULL, a diagnostic client that will be attached to
      /// (and optionally, depending on the shouldOwnClient parameter, owned by) the
      /// returned DiagnosticsEngine object.
      ///
      /// @return the new object on success, or null on failure.
      static clang::IntrusiveRefCntPtr<clang::DiagnosticsEngine> createDiagnostics(
          clang::DiagnosticOptions* opts,
          clang::DiagnosticConsumer* client = nullptr,
          bool shouldOwnClient = true);

      void createDiagnostics(clang::DiagnosticConsumer* client = nullptr, bool shouldOwnClient = true);

      /// Get the current diagnostics engine.
      clang::DiagnosticsEngine& getDiagnostics() const;

      /// Get the current diagnostics client.
      clang::DiagnosticConsumer& getDiagnosticClient() const;

      /// Get the frontend options.
      FrontendOptions& getFrontendOptions();

      /// Get the frontend options.
      const FrontendOptions& getFrontendOptions() const;

      /// Get the code generation options.
      CodegenOptions& getCodegenOptions();

      /// Get the code generation options.
      const CodegenOptions& getCodegenOptions() const;

      /// Get the simulation options.
      SimulationOptions& getSimulationOptions();

      /// Get the simulation options.
      const SimulationOptions& getSimulationOptions() const;

      /// Get the diagnostic options.
      clang::DiagnosticOptions& getDiagnosticOptions();

      /// Get the diagnostic options.
      const clang::DiagnosticOptions& getDiagnosticOptions() const;

      /// Execute a frontend action.
      ///
      /// @param action the action to be executed
      /// @return whether the execution has been successful or not
      bool executeAction(FrontendAction& action);

      /// Clear the output file list.
      ///
      /// @param eraseFiles   whether the registered output files should be erased or not
      void clearOutputFiles(bool eraseFiles);

      /// Get the MLIR context.
      mlir::MLIRContext& getMLIRContext();

      /// Get the MLIR context.
      const mlir::MLIRContext& getMLIRContext() const;

      /// Get the LLVM context.
      llvm::LLVMContext& getLLVMContext();

      /// Get the LLVM context.
      const llvm::LLVMContext& getLLVMContext() const;

      std::string& getFlattened();

      const std::string& getFlattened() const;

      void setFlattened(std::string value);

      std::unique_ptr<ast::Class>& getAST();

      const std::unique_ptr<ast::Class>& getAST() const;

      void setAST(std::unique_ptr<ast::Class> value);

      mlir::ModuleOp& getMLIRModule();

      const mlir::ModuleOp& getMLIRModule() const;

      void setMLIRModule(std::unique_ptr<mlir::ModuleOp> module);

      llvm::Module& getLLVMModule();

      const llvm::Module& getLLVMModule() const;

      void setLLVMModule(std::unique_ptr<llvm::Module> module);

      /// Create the default output file (based on the invocation's options) and
      /// add it to the list of tracked output files. If the name of the output
      /// file is not provided, it will be derived from the input file.
      ///
      /// @param binary     the mode to open the file in.
      /// @param baseInput  if the invocation contains no output file name (i.e.
      ///                   outputFile in FrontendOptions is empty), the input path
      ///                   name to use for deriving the output path.
      /// @param extension  the extension to use for output names derived from
      ///                   the baseInput parameter.
      /// @return null on error, ostream for the output file otherwise
      std::unique_ptr<llvm::raw_pwrite_stream> createDefaultOutputFile(
          bool binary = true, llvm::StringRef baseInput = "", llvm::StringRef extension = "");

    private:
      /// Create a new output file
      ///
      /// @param outputPath   the path to the output file.
      /// @param binary       the mode to open the file in.
      /// @return null on error, ostream for the output file otherwise
      llvm::Expected<std::unique_ptr<llvm::raw_pwrite_stream>> createOutputFileImpl(
          llvm::StringRef outputPath, bool binary);

    private:
      /// The options used in this compiler instance
      std::shared_ptr<CompilerInvocation> invocation_;

      /// The diagnostics engine instance
      llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> diagnostics_;

      /// The list of active output files
      std::list<OutputFile> outputFiles_;

      /// Holds the output stream provided by the user. Normally, users of
      /// CompilerInstance will call CreateOutputFile to obtain/create an output
      /// stream. If they want to provide their own output stream, this field will
      /// facilitate this. It is optional and will normally be just a nullptr.
      std::unique_ptr<llvm::raw_pwrite_stream> outputStream_;

      std::unique_ptr<mlir::MLIRContext> mlirContext_;
      std::unique_ptr<llvm::LLVMContext> llvmContext_;

      std::string flattened_;
      std::unique_ptr<ast::Class> ast_;
      std::unique_ptr<mlir::ModuleOp> mlirModule_;
      std::unique_ptr<llvm::Module> llvmModule_;
  };

  /// Construct the FrontendAction of a compiler invocation based on the
  /// options specified for the compiler invocation.
  ///
  /// @return the created FrontendAction object
  std::unique_ptr<FrontendAction> createFrontendAction(CompilerInstance& ci);

  /// Execute the given actions described by the compiler invocation object
  /// in the given compiler instance.
  ///
  /// @return true on success; false otherwise
  bool executeCompilerInvocation(CompilerInstance* instance);
}

#endif // MARCO_FRONTEND_COMPILERINSTANCE_H
