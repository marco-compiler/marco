#ifndef MARCO_FRONTEND_DIAGNOSTICHANDLER_H
#define MARCO_FRONTEND_DIAGNOSTICHANDLER_H

#include "marco/Frontend/CompilerInstance.h"
#include "mlir/IR/Diagnostics.h"
#include "clang/Basic/Diagnostic.h"

namespace marco::frontend
{
  class DiagnosticHandler
  {
    public:
      DiagnosticHandler(CompilerInstance& instance);

      using NoteVector = std::vector<std::unique_ptr<mlir::Diagnostic>>;
      using note_iterator = llvm::pointee_iterator<NoteVector::iterator>;

      mlir::LogicalResult emit(
          mlir::DiagnosticSeverity severity,
          mlir::Location loc,
          llvm::StringRef message,
          llvm::ArrayRef<std::string> notes);

      mlir::LogicalResult emit(
          mlir::DiagnosticSeverity severity,
          mlir::Location loc,
          llvm::StringRef message);

    private:
      CompilerInstance* instance;
  };
}

#endif // MARCO_FRONTEND_DIAGNOSTICHANDLER_H