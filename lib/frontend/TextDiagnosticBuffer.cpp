#include <clang/Basic/Diagnostic.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/ErrorHandling.h>
#include <marco/frontend/TextDiagnosticBuffer.h>

namespace marco::frontend
{
  void TextDiagnosticBuffer::HandleDiagnostic(
      clang::DiagnosticsEngine::Level level, const clang::Diagnostic& info)
  {
    DiagnosticConsumer::HandleDiagnostic(level, info);

    llvm::SmallString<100> buf;
    info.FormatDiagnostic(buf);

    switch (level) {
      default:
        llvm_unreachable("Diagnostic not handled during diagnostic buffering!");

      case clang::DiagnosticsEngine::Note:
        all.emplace_back(level, notes.size());
        notes.emplace_back(info.getLocation(), std::string(buf.str()));
        break;

      case clang::DiagnosticsEngine::Warning:
        all.emplace_back(level, warnings.size());
        warnings.emplace_back(info.getLocation(), std::string(buf.str()));
        break;

      case clang::DiagnosticsEngine::Remark:
        all.emplace_back(level, remarks.size());
        remarks.emplace_back(info.getLocation(), std::string(buf.str()));
        break;

      case clang::DiagnosticsEngine::Error:
      case clang::DiagnosticsEngine::Fatal:
        all.emplace_back(level, errors.size());
        errors.emplace_back(info.getLocation(), std::string(buf.str()));
        break;
    }
  }

  void TextDiagnosticBuffer::FlushDiagnostics(clang::DiagnosticsEngine& Diags) const
  {
    for (const auto& i: all) {
      auto Diag = Diags.Report(Diags.getCustomDiagID(i.first, "%0"));

      switch (i.first) {
        default:
          llvm_unreachable("Diagnostic not handled during diagnostic flushing!");

        case clang::DiagnosticsEngine::Note:
          Diag << notes[i.second].second;
          break;

        case clang::DiagnosticsEngine::Warning:
          Diag << warnings[i.second].second;
          break;

        case clang::DiagnosticsEngine::Remark:
          Diag << remarks[i.second].second;
          break;

        case clang::DiagnosticsEngine::Error:
        case clang::DiagnosticsEngine::Fatal:
          Diag << errors[i.second].second;
          break;
      }
    }
  }
}
