#ifndef MARCO_FRONTEND_COMPILERINVOCATION_H
#define MARCO_FRONTEND_COMPILERINVOCATION_H

#include "marco/Frontend/CodegenOptions.h"
#include "marco/Frontend/FrontendOptions.h"
#include "marco/Frontend/LanguageOptions.h"
#include "marco/Frontend/SimulationOptions.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <memory>

namespace marco::frontend {
class CompilerInvocationBase {
private:
  /// Options controlling the language variant.
  llvm::IntrusiveRefCntPtr<LanguageOptions> languageOptions;

  /// Options controlling the target.
  std::shared_ptr<clang::TargetOptions> targetOptions;

  /// Options controlling the diagnostic engine.
  std::shared_ptr<clang::DiagnosticOptions> diagnosticOptions;

  /// Options controlling file system operations.
  std::shared_ptr<clang::FileSystemOptions> fileSystemOptions;

  /// Options controlling the frontend.
  std::shared_ptr<FrontendOptions> frontendOptions;

  std::shared_ptr<CodegenOptions> codegenOptions;
  std::shared_ptr<SimulationOptions> simulationOptions;

protected:
  /// Dummy tag type whose instance can be passed into the constructor to
  /// prevent creation of the reference-counted option objects.
  struct EmptyConstructor {};

  CompilerInvocationBase();

  CompilerInvocationBase(EmptyConstructor);

  CompilerInvocationBase(const CompilerInvocationBase &other) = delete;

  CompilerInvocationBase(CompilerInvocationBase &&other);

  CompilerInvocationBase &
  operator=(const CompilerInvocationBase &other) = delete;

  CompilerInvocationBase &deepCopyAssign(const CompilerInvocationBase &other);

  CompilerInvocationBase &
  shallowCopyAssign(const CompilerInvocationBase &other);

  CompilerInvocationBase &operator=(CompilerInvocationBase &&other);

  ~CompilerInvocationBase();

public:
  /// @name Getters.
  /// {

  LanguageOptions &getLanguageOptions();

  const LanguageOptions &getLanguageOptions() const;

  clang::TargetOptions &getTargetOptions();

  const clang::TargetOptions &getTargetOptions() const;

  std::shared_ptr<clang::TargetOptions> getTargetOptionsPtr();

  clang::DiagnosticOptions &getDiagnosticOptions();

  const clang::DiagnosticOptions &getDiagnosticOptions() const;

  clang::FileSystemOptions &getFileSystemOptions();

  const clang::FileSystemOptions &getFileSystemOptions() const;

  FrontendOptions &getFrontendOptions();

  const FrontendOptions &getFrontendOptions() const;

  CodegenOptions &getCodeGenOptions();

  const CodegenOptions &getCodeGenOptions() const;

  SimulationOptions &getSimulationOptions();

  const SimulationOptions &getSimulationOptions() const;

  /// }
};

class CompilerInvocation : public CompilerInvocationBase {
public:
  /// Create a compiler invocation from a list of input options.
  static bool createFromArgs(CompilerInvocation &res,
                             llvm::ArrayRef<const char *> commandLineArgs,
                             clang::DiagnosticsEngine &diagnostics);
};

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
createVFSFromCompilerInvocation(const CompilerInvocation &ci,
                                clang::DiagnosticsEngine &diags);

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> createVFSFromCompilerInvocation(
    const CompilerInvocation &ci, clang::DiagnosticsEngine &diags,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> baseFS);
} // namespace marco::frontend

#endif // MARCO_FRONTEND_COMPILERINVOCATION_H
