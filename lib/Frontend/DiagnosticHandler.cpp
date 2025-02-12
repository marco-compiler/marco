#include "marco/Frontend/DiagnosticHandler.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco::frontend;

namespace marco::frontend {
DiagnosticHandler::DiagnosticHandler(CompilerInstance &instance)
    : instance(&instance) {}

mlir::LogicalResult DiagnosticHandler::emit(mlir::DiagnosticSeverity severity,
                                            mlir::Location loc,
                                            llvm::StringRef message,
                                            llvm::ArrayRef<std::string> notes) {
  if (mlir::failed(emit(severity, loc, message))) {
    return mlir::failure();
  }

  for (const auto &note : notes) {
    if (mlir::failed(emit(mlir::DiagnosticSeverity::Note, loc, note))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult DiagnosticHandler::emit(mlir::DiagnosticSeverity severity,
                                            mlir::Location loc,
                                            llvm::StringRef message) {
  // Determine the clang-equivalent severity.
  clang::DiagnosticsEngine::Level level;

  if (severity == mlir::DiagnosticSeverity::Error) {
    level = clang::DiagnosticsEngine::Error;
  } else if (severity == mlir::DiagnosticSeverity::Warning) {
    level = clang::DiagnosticsEngine::Level::Warning;
  } else if (severity == mlir::DiagnosticSeverity::Note) {
    level = clang::DiagnosticsEngine::Level::Note;
  } else if (severity == mlir::DiagnosticSeverity::Remark) {
    level = clang::DiagnosticsEngine::Level::Remark;
  } else {
    return mlir::failure();
  }

  auto &sourceManager = instance->getSourceManager();

  if (auto fileLineColLoc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
    // Convert the MLIR location to a Clang location.
    auto fileRef =
        sourceManager.getFileManager().getFileRef(fileLineColLoc.getFilename());

    if (!fileRef) {
      llvm::consumeError(fileRef.takeError());
      return mlir::failure();
    }

    // Emit the message.
    auto &diags = instance->getDiagnostics();

    if (fileLineColLoc.getLine() != 0) {
      auto clangLoc = sourceManager.translateFileLineCol(
          *fileRef, fileLineColLoc.getLine(), fileLineColLoc.getColumn());

      diags.Report(clangLoc, diags.getCustomDiagID(level, "%0")) << message;
    } else {
      diags.Report(diags.getCustomDiagID(level, "%0")) << message;
    }

    return mlir::success();
  }

  return mlir::failure();

  /*
  clang::TextDiagnostic textDiagnostic(
      OS, Ctx.getLangOpts(), &Diags.getDiagnosticOptions());

  textDiagnostic.emitDiagnostic(
      fullSourceLoc, level, message,
      clang::CharSourceRange::getTokenRange(R),
      clang::FixItHint::CreateInsertion(clangLoc, note));
      */
}
} // namespace marco::frontend
